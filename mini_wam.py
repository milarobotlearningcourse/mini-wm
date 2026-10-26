## mini-WAM: a small world action model. It is mini-FM's transformer (mini_fm.py), on robot data only, that predicts the
## next policy.action_stacking actions (from its class token, with mini-FM's action head) and also the next
## policy.action_stacking camera frames, at model.frame_size pixels, by diffusion: the noisy patches of the future
## frames are extra tokens at the end of the sequence, which the same transformer blocks learn to denoise. The attention
## mask keeps the class token (and every token before it) from seeing them, so acting runs only mini-FM's part of the
## sequence and costs the same as mini-FM; predicting the future only shapes the features the actions are read from.
## mini_train.py trains it (--config-name mini-wam).
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from mini_fm import (plain_config, get_patches_fast, calc_positional_embeddings, cosine_noise_schedule, Block,
                     MSEActionHead, DiffusionActionHead, FMPolicy)


class WAM(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self._cfg = cfg
    self._plain_cfg = plain_config(cfg) ## what forward() reads, so the compiled forward is one graph
    n_patches = int(cfg.image_shape[0] / cfg.patch_size) * int(cfg.image_shape[1] / cfg.patch_size)
    assert cfg.image_block_size == n_patches, f"image_block_size must be {n_patches} (one token per patch), got {cfg.image_block_size}"
    ## mini-FM's sequence: the pose token (with policy.use_pose_data), the image tokens, the goal text and the class token
    self.n_obs = int(cfg.policy.use_pose_data) + cfg.image_block_size + cfg.max_block_size + 1
    ## then the patches of the next action_stacking frames, each downsampled to frame_size x frame_size and cut into
    ## frame_patch_size patches (smaller than the image's patch_size: more tokens, each with fewer pixels to predict)
    self.n_frames = cfg.policy.action_stacking
    self.frame_tokens = self.n_frames * (cfg.model.frame_size // cfg.model.frame_patch_size) ** 2
    self.frame_d = 3 * cfg.model.frame_patch_size ** 2 ## the pixels of one frame patch
    context_length = self.n_obs + self.frame_tokens
    self.register_buffer('positional_embeddings', calc_positional_embeddings(context_length, cfg.n_embd), persistent=False)
    ## True where a token may attend. Over its own tokens, mini-FM's mask: the pose and image tokens see all of them, the
    ## text and the class token are causal. None of them see the frame tokens, which see everything.
    mask = torch.ones(context_length, context_length, dtype=torch.bool)
    prefix = self.n_obs - cfg.max_block_size - 1
    mask[:self.n_obs, self.n_obs:] = False
    mask[prefix:self.n_obs, :self.n_obs] = torch.tril(mask[prefix:self.n_obs, :self.n_obs], diagonal=prefix)
    self.register_buffer('mask', mask, persistent=False)

    self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
    self.class_tokens = nn.Parameter(torch.rand(1, cfg.n_embd))
    self.input_d = int(cfg.image_shape[2] * cfg.patch_size * cfg.patch_size)
    self.lin_map = nn.Linear(self.input_d, cfg.n_embd, bias=False)
    if cfg.policy.use_pose_data: ## one pose token before the image tokens
        self.lin_map_pose = nn.Linear(cfg.pose_dim, cfg.n_embd, bias=False)
    self.blocks = nn.ModuleList([Block(cfg.n_embd, cfg.n_head, dropout=cfg.dropout, cfg=cfg) for _ in range(cfg.n_blocks)])
    ## Continuous action chunks: policy.action_stacking actions of action_dim
    self.action_head = (MSEActionHead(nn.Linear(cfg.n_embd, cfg.action_dim * cfg.policy.action_stacking))
                        if cfg.model.action_head != "diffusion" else DiffusionActionHead(cfg)) ## MSE: huber and l1 are mini-FM's

    ## The future frames' DDPM, with the action diffusion head's noise schedule: each frame token is a noisy patch,
    ## plus the embedding of its noise step, and predicts the clean patch. Not the noise, as DiffusionActionHead does:
    ## at the noisiest steps the noisy patch is nearly all noise, so predicting the noise is easy but says nothing about
    ## the frames, which come out as unrelated patches; predicting them makes the model work them out from the current
    ## image and goal.
    self.steps = cfg.model.diffusion_steps
    betas, alpha_bar = cosine_noise_schedule(self.steps)
    self.register_buffer("betas", betas, persistent=False)
    self.register_buffer("alpha_bar", alpha_bar, persistent=False)
    self.step_embedding = nn.Embedding(self.steps, cfg.n_embd)
    self.lin_map_frames = nn.Linear(self.frame_d, cfg.n_embd)
    self.frame_out = nn.Linear(cfg.n_embd, self.frame_d)
    if cfg.model.frame_adaln:
        ## adaLN-Zero (DiT, Peebles & Xie 2023): in each block, the noise step sets a shift, scale and gate of the frame
        ## tokens' attention and feed-forward. The gates start at 0, so each block starts as the identity for them.
        ## The other tokens are not modulated, so acting is unchanged.
        self.frame_modulation = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(cfg.n_embd, 6 * cfg.n_embd)) for _ in self.blocks])
        for modulation in self.frame_modulation:
            nn.init.zeros_(modulation[1].weight)
            nn.init.zeros_(modulation[1].bias)
        self.register_buffer("is_frame", (torch.arange(context_length) >= self.n_obs).float()[:, None], persistent=False)

  def frame_patches(self, frames):
    """ Frames stacked in the channels, (B, H, W, n * 3) as RobotCache's future frames, downsampled to frame_size and
    cut into patches: (B, n * patches per frame, input_d). """
    cfg = self._plain_cfg
    small = F.avg_pool2d(frames.permute(0, 3, 1, 2), frames.shape[1] // cfg.model.frame_size) ## (B, n * 3, frame_size, frame_size)
    p = cfg.model.frame_patch_size
    return rearrange(small, 'b (f c) (h p1) (w p2) -> b (f h w) (p1 p2 c)', c=3, p1=p, p2=p)

  def patches_to_frames(self, patches):
    """ The inverse of frame_patches: (B, n, frame_size, frame_size, 3). """
    cfg = self._plain_cfg
    p = cfg.model.frame_patch_size
    return rearrange(patches, 'b (f h w) (p1 p2 c) -> b f (h p1) (w p2) c', f=self.n_frames, h=cfg.model.frame_size // p, p1=p, p2=p)

  def features(self, X_text, X_images, pose, noisy_frames=None, t=None):
    """ The transformer's output tokens for mini-FM's sequence, followed by the tokens of the noisy future frame patches
    at noise step t, if given, and what the action head reads: the class token's features, or with the diffusion head
    each layer's keys and values of mini-FM's sequence (not the frames'), which its action expert attends to. """
    cfg = self._plain_cfg
    n = len(X_images)
    out = self.lin_map(get_patches_fast(X_images, cfg))
    if cfg.policy.use_pose_data:
        out = torch.cat((self.lin_map_pose(pose)[:, None], out), dim=1)
    out = torch.cat((out, self.token_embedding_table(X_text), self.class_tokens.expand(n, 1, -1)), dim=1)
    if noisy_frames is not None:
        out = torch.cat((out, self.lin_map_frames(noisy_frames) + self.step_embedding(t)[:, None]), dim=1)
    T = out.shape[1]
    out = out + self.positional_embeddings[:T]
    prefix = []
    for i, block in enumerate(self.blocks):
        if cfg.model.action_head == "diffusion":
            prefix.append(block.sa.keys_values(block.ln1(out[:, :self.n_obs])))
        out = block(out, self.mask[:T, :T], self.modulation(i, t, T) if noisy_frames is not None and cfg.model.frame_adaln else None)
    return out, ((torch.stack(prefix), None) if prefix else out[:, self.n_obs - 1])

  def modulation(self, i, t, T):
    """ Block i's adaLN-Zero shift, scale and gate of the attention and the feed-forward, each (B, T, n_embd): from
    noise step t for the frame tokens, and 0, 0 and 1 (no change) for the others. """
    is_frame = self.is_frame[:T]
    shift1, scale1, gate1, shift2, scale2, gate2 = (is_frame * m[:, None] for m in self.frame_modulation[i](self.step_embedding(t)).chunk(6, dim=-1))
    return shift1, scale1, 1 - is_frame + gate1, shift2, scale2, 1 - is_frame + gate2

  def forward(self, X_text, X_images, targets=None, actions=None, pose=None, frames=None):
    """ FM.forward for robot rows (targets, all -1, are not used), plus the frame diffusion's loss. Returns the class
    token's features when targets is False (from act()), otherwise (None, loss, parts). """
    if targets is False:
        return self.features(X_text, X_images, pose)[1]
    cfg = self._plain_cfg
    ## Noise each row's future frames to a random step; the frame tokens learn to predict the clean frames
    clean = self.frame_patches(frames)
    t = torch.randint(0, self.steps, (len(clean),), device=clean.device)
    alpha_bar = self.alpha_bar[t, None, None]
    out, cond = self.features(X_text, X_images, pose, alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * torch.randn_like(clean), t)
    parts = {self.action_head.loss_name: self.action_head.loss(cond, actions).mean(),
             "frames": F.mse_loss(self.frame_out(out[:, self.n_obs:]).float(), clean)} ## in pixels
    ## parts stay unweighted for logging
    loss = cfg.mse_head_weights * parts[self.action_head.loss_name] + cfg.model.frame_loss_weight * parts["frames"]
    return (None, loss, parts)

  def act(self, X_text, X_images, pose):
    """ The (whitened) action chunk predicted for each row, flattened. """
    return self.action_head.sample(self(X_text, X_images, targets=False, pose=pose))

  @torch.no_grad()
  def predict_frames(self, X_text, X_images, pose):
    """ Samples the next n frames, (B, n, frame_size, frame_size, 3): starts from pure noise and removes it one step at
    a time. Each step predicts the clean frames, clamps them to the pixel range, then moves to the DDPM posterior's
    mean between them and x (Ho et al. 2020). """
    low, high = self._cfg.image_bounds
    x = torch.randn(len(X_images), self.frame_tokens, self.frame_d, device=X_images.device)
    for t in reversed(range(self.steps)):
        out, _ = self.features(X_text, X_images, pose, x, torch.full((len(x),), t, device=x.device))
        clean = self.frame_out(out[:, self.n_obs:]).float().clamp(low, high)
        beta, alpha_bar = self.betas[t], self.alpha_bar[t]
        alpha_bar_prev = self.alpha_bar[t - 1] if t > 0 else torch.ones_like(alpha_bar)
        x = (alpha_bar_prev.sqrt() * beta * clean + (1 - beta).sqrt() * (1 - alpha_bar_prev) * x) / (1 - alpha_bar)
        if t > 0: ## every step but the last adds back a little noise
            x = x + (beta * (1 - alpha_bar_prev) / (1 - alpha_bar)).sqrt() * torch.randn_like(x)
    return self.patches_to_frames(x)

  def frame_grid(self, batch, rows=4):
    """ An image of the predicted future, for logging: for each of the first rows samples of a batch, its current frame,
    its real next frames, then the predicted ones, all at frame_size. (H, W, 3) uint8. """
    X_text, X_images, _, _, pose, frames = (x[:rows] for x in batch)
    current = self.patches_to_frames(self.frame_patches(X_images.repeat(1, 1, 1, self.n_frames)))[:, :1]
    tiles = torch.cat((current, self.patches_to_frames(self.frame_patches(frames)), self.predict_frames(X_text, X_images, pose)), dim=1)
    low, high = self._cfg.image_bounds
    grid = rearrange(tiles.float(), 'r f h w c -> (r h) (f w) c')
    return ((grid - low) / (high - low) * 255).clamp(0, 255).byte().cpu().numpy()


def make_model(cfg):
  return WAM(cfg)


def make_policy(cfg): ## for BackgroundSimEval, which builds its own copy of the policy
    return FMPolicy(make_model(cfg), cfg)

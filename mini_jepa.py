## mini-JEPA: a small latent world model, after RoboJEPA (arXiv 2610.10515), on robot data: from the current camera frame
## and the next model.sequence_steps actions taken, it predicts the embeddings of the frames that follow, never their
## pixels. A small ViT (the encoder) turns each frame into one embedding per patch. The predictor, a transformer, reads
## each timestep as one action token (the next model.frame_skip actions) and the frame's patch embeddings, with
## timestep-causal attention (a token sees its own timestep and earlier ones), and predicts the next frame's embeddings
## at the frame's patch tokens. As RoboJEPA, it learns with an L1 loss on the embeddings, half teacher-forced (each step
## from the real frames before it) and half autoregressive (from the current frame only, feeding its own predictions
## back: the error of these rollouts is RoboJEPA's imagination error). RoboJEPA's encoder is a frozen V-JEPA 2.1 (1B
## parameters); there is no small one, so here the encoder learns too, with the targets from an exponential moving
## average of it (update_target, after every optimizer step), as V-JEPA's own pre-training: predicting a slowly moving
## copy keeps the embeddings from collapsing to a constant, which would make the loss 0. Left out of RoboJEPA: the state
## token and RoPE (learned and sinusoidal position embeddings instead). model.policy picks how it acts in the sim evals:
## - cem, as RoboJEPA: plan with the cross-entropy method, searching for the actions whose imagined last frame's
##   embedding is nearest the goal image's. Like RoboJEPA it never reads the goal text. The sim gives no goal image, so
##   each episode starts from the init state of one of the dataset's demos and its goal is that demo's last frame
##   (sim_eval.dataset_goal_images);
## - fm: mini-FM (mini_fm.py), trained next to the world model as in mini_dreamer.py, is the policy.
## mini_train.py trains it (--config-name mini-jepa).
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from mini_fm import plain_config, calc_positional_embeddings, Block, FM, FMPolicy


class Encoder(nn.Module):
  """ A small ViT: the patches of a frame, (N, patches, 3 * patch_size^2), to one embedding per patch, (N, patches,
  n_embd), each normalized (LayerNorm without its scale and shift, as V-JEPA's targets). """
  def __init__(self, cfg, n_patches):
    super().__init__()
    self.patch_embedding = nn.Linear(3 * cfg.model.frame_patch_size ** 2, cfg.n_embd)
    self.register_buffer("positions", calc_positional_embeddings(n_patches, cfg.n_embd), persistent=False)
    self.register_buffer("mask", torch.ones(n_patches, n_patches, dtype=torch.bool), persistent=False) ## every patch sees every patch
    self.blocks = nn.ModuleList([Block(cfg.n_embd, cfg.n_head, dropout=0.0, cfg=cfg) for _ in range(cfg.model.encoder_blocks)])

  def forward(self, patches):
    x = self.patch_embedding(patches) + self.positions
    for block in self.blocks:
        x = block(x, self.mask)
    return F.layer_norm(x, x.shape[-1:])


class JEPA(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self._cfg = cfg
    self._plain_cfg = plain_config(cfg) ## what forward() reads, so the compiled forward is one graph
    m = cfg.model
    assert m.policy in ("cem", "fm"), m.policy
    if m.policy == "fm":
        self.fm = FM(cfg) ## the policy
    assert m.sequence_steps % m.frame_skip == 0, f"sequence_steps ({m.sequence_steps}) must be a multiple of frame_skip ({m.frame_skip})"
    self.skip, self.n_steps = m.frame_skip, m.sequence_steps // m.frame_skip ## each timestep is frame_skip actions
    self.n_patches = (m.frame_size // m.frame_patch_size) ** 2
    self.encoder = Encoder(cfg, self.n_patches)
    self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False) ## the moving average of the encoder
    ## The predictor. Each timestep is an action token, then the frame's patch tokens: tokens of timestep t attend to
    ## those of timesteps <= t, and its patch tokens predict the patches' embeddings at timestep t + 1
    self.action_embedding = nn.Linear(m.frame_skip * cfg.action_dim, cfg.n_embd)
    self.predictor_in = nn.Linear(cfg.n_embd, cfg.n_embd)
    tokens = 1 + self.n_patches
    self.register_buffer("token_positions", calc_positional_embeddings(tokens, cfg.n_embd), persistent=False)
    self.time_embedding = nn.Embedding(self.n_steps, cfg.n_embd)
    timestep = torch.arange(self.n_steps).repeat_interleave(tokens)
    self.register_buffer("mask", timestep[None, :] <= timestep[:, None], persistent=False) ## True where token [row] may attend to [column]
    self.predictor = nn.ModuleList([Block(cfg.n_embd, cfg.n_head, dropout=0.0, cfg=cfg) for _ in range(m.predictor_blocks)])
    self.predictor_out = nn.Linear(cfg.n_embd, cfg.n_embd)
    self.goal_conditioned = m.policy == "cem" ## FMPolicy then gives the sim's goal image to plan_to, not the goal text to act
    self.chunk = cfg.policy.action_stacking * cfg.action_dim

  @torch.no_grad()
  def update_target(self):
    """ Moves the target encoder towards the encoder: target = ema * target + (1 - ema) * encoder. Called by
    mini_train.py after each optimizer step. """
    target, online = list(self.target_encoder.parameters()), list(self.encoder.parameters())
    torch._foreach_lerp_(target, online, 1 - self._plain_cfg.model.ema)

  def patchify(self, images):
    """ Frames stacked in the channels, (B, H, W, F * 3), downsampled to frame_size and cut into patches:
    (B, F, patches, 3 * patch_size^2). """
    m = self._plain_cfg.model
    images = images.permute(0, 3, 1, 2)
    small = F.avg_pool2d(images, images.shape[-1] // m.frame_size).unflatten(1, (-1, 3)) ## (B, F, 3, size, size)
    return rearrange(small, 'b f c (h p1) (w p2) -> b f (h w) (p1 p2 c)', p1=m.frame_patch_size, p2=m.frame_patch_size)

  def observations(self, X_images, frames):
    """ The patches of the current frame and of the frame after every frame_skip-th of the next sequence_steps actions:
    (B, 1 + n_steps, patches, 3 * patch_size^2). X_images (B, H, W, 3), frames (B, H, W, sequence_steps * 3) stacked
    in the channels, as RobotCache's future frames. """
    frames = frames.unflatten(-1, (-1, 3))[..., self.skip - 1::self.skip, :].flatten(-2) ## the frame after action skip - 1, 2 * skip - 1, ...
    return self.patchify(torch.cat((X_images, frames), dim=-1))

  def encode(self, encoder, observations):
    """ observations (B, F, patches, d) to their embeddings, (B, F, patches, n_embd). """
    return encoder(observations.flatten(0, 1)).unflatten(0, observations.shape[:2])

  def predict(self, z, actions):
    """ The predicted embeddings of the frame after each timestep, (B, k, patches, n_embd), from the embeddings of
    k frames, z (B, k, patches, n_embd), and the actions after each, (B, k, frame_skip * action_dim). """
    k = z.shape[1]
    x = torch.cat((self.action_embedding(actions)[:, :, None], self.predictor_in(z)), dim=2) ## (B, k, 1 + patches, n_embd)
    x = (x + self.token_positions + self.time_embedding.weight[:k, None]).flatten(1, 2)
    T = x.shape[1]
    for block in self.predictor:
        x = block(x, self.mask[:T, :T])
    out = self.predictor_out(x.unflatten(1, (k, -1))[:, :, 1:]) ## the patch tokens
    return F.layer_norm(out, out.shape[-1:])

  def rollout(self, z, actions):
    """ Imagines the embeddings of the frames after each of the timesteps' actions (B, n, frame_skip * action_dim),
    from the current frame's, z (B, 1, patches, n_embd): each step predicts the next frame from the frames predicted
    so far. (B, n, patches, n_embd). """
    for k in range(actions.shape[1]):
        z = torch.cat((z, self.predict(z, actions[:, :k + 1])[:, -1:]), dim=1)
    return z[:, 1:]

  def forward(self, X_text, X_images, targets=None, actions=None, pose=None, frames=None, sequence_actions=None):
    """ The world model's loss on the embeddings of the frames after the next sequence_steps actions taken (frames,
    after each of sequence_actions), plus with model.policy fm FM.forward's (the policy's action loss on the chunk,
    actions). Returns (logits, loss, parts), or with fm the class token's features when targets is False. """
    world_model, world_parts = self.world_model_loss(X_images, sequence_actions, frames)
    if self._plain_cfg.model.policy == "cem":
        return None, world_model, world_parts
    if targets is False:
        return self.fm(X_text, X_images, targets=False, pose=pose)
    logits, fm_loss, parts = self.fm(X_text, X_images, targets, actions, pose)
    return (logits, fm_loss + self._plain_cfg.model.frame_loss_weight * world_model, {**parts, **world_parts})

  ## Run eagerly, outside the training's compiled graph, as mini_dreamer.py's (in bf16, torch 2.5.1's compiled backward
  ## of mini-FM and a world model in one graph can turn mini-FM's gradients to NaN)
  @torch.compiler.disable
  def world_model_loss(self, X_images, actions, frames):
    """ RoboJEPA's loss, the mean of the teacher-forced and the autoregressive L1 errors of the predicted embeddings,
    and its parts: those two (latent_ar is the imagination error), latent_copy (the error of predicting that nothing
    changes, each frame's embedding as the next one's: the model should beat it) and target_std (the targets' std over
    the batch: near 0 when the embeddings collapse to a constant). """
    observations = self.observations(X_images, frames)
    actions = actions.unflatten(1, (self.n_steps, -1)) ## (B, n_steps, frame_skip * action_dim)
    with torch.no_grad():
        targets = self.encode(self.target_encoder, observations).float() ## (B, 1 + n_steps, patches, n_embd)
    context = self.encode(self.encoder, observations[:, :-1])
    parts = {"latent_tf": (self.predict(context, actions).float() - targets[:, 1:]).abs().mean(),
             "latent_ar": (self.rollout(context[:, :1], actions).float() - targets[:, 1:]).abs().mean(),
             "latent_copy": (targets[:, :-1] - targets[:, 1:]).abs().mean(),
             "target_std": targets.std(0).mean()}
    return 0.5 * parts["latent_tf"] + 0.5 * parts["latent_ar"], parts

  @torch.no_grad()
  def plan(self, z, goal):
    """ The cross-entropy method, as RoboJEPA: the first action chunk of the action sequence (n_steps timesteps) whose
    imagined last frame is nearest (the L1 error of the embeddings) the goal image, from each row's current frame's
    embeddings z (B, 1, patches, n_embd) and the goal's, (B, patches, n_embd). The actions are whitened, as trained on,
    so the search starts from the standard normal. (B, chunk). """
    m = self._plain_cfg.model
    B, N = len(z), m.cem_samples
    mean = torch.zeros(B, self.n_steps, self.action_embedding.in_features, device=z.device)
    std = torch.ones_like(mean)
    z, goal, rows = z.repeat_interleave(N, 0), goal.repeat_interleave(N, 0), torch.arange(B, device=z.device)[:, None]
    for _ in range(m.cem_iterations):
        actions = mean[:, None] + std[:, None] * torch.randn(B, N, *mean.shape[1:], device=z.device) ## (B, N, n_steps, skip * action_dim)
        cost = (self.rollout(z, actions.flatten(0, 1))[:, -1] - goal).abs().mean((1, 2)).unflatten(0, (B, N))
        elites = actions[rows, cost.topk(m.cem_elites, largest=False).indices] ## the cem_elites nearest the goal
        mean, std = elites.mean(1), elites.std(1)
    return mean.flatten(1)[:, :self.chunk]

  def act(self, X_text, X_images, pose):
    """ mini-FM's (whitened) action chunk for each row, flattened (fm). """
    return self.fm.act(X_text, X_images, pose)

  def plan_to(self, X_images, goal_images):
    """ The (whitened) action chunk for each row, flattened, planned from its frame X_images (B, H, W, 3) towards its
    goal image (B, H, W, 3), both scaled as the inputs (cem). """
    z = self.encode(self.encoder, self.patchify(X_images))
    return self.plan(z, self.encode(self.encoder, self.patchify(goal_images))[:, 0])

  @torch.no_grad()
  def frame_grid(self, batch, rows=4):
    """ An image of the imagined future, for logging. The model predicts embeddings, not pixels, so each imagined
    frame is shown as the frame of the batch (any sample, any timestep) whose embedding is nearest (L1): when the
    rollout is right it is the real frame. For each of the first rows samples: its current frame, its real next frames,
    then the nearest ones to the rollout's embeddings, all at frame_size. (H, W, 3) uint8. """
    X_text, X_images, _, _, pose, frames, actions = batch
    observations = self.observations(X_images, frames)
    pool = self.encode(self.target_encoder, observations).flatten(0, 1).flatten(1).float() ## (B * (1 + n_steps), patches * n_embd)
    z = self.encode(self.encoder, observations[:rows, :1])
    imagined = self.rollout(z, actions[:rows].unflatten(1, (self.n_steps, -1))).flatten(2).float() ## (rows, n_steps, patches * n_embd)
    nearest = torch.cdist(imagined.flatten(0, 1), pool, p=1).argmin(-1)
    tiles = torch.cat((observations[:rows], observations.flatten(0, 1)[nearest].unflatten(0, (rows, -1))), dim=1)
    p, low, high = self._plain_cfg.model.frame_patch_size, *self._cfg.image_bounds
    grid = rearrange(tiles.float(), 'r f (h w) (p1 p2 c) -> (r h p1) (f w p2) c', h=self._plain_cfg.model.frame_size // p, p1=p, p2=p)
    return ((grid - low) / (high - low) * 255).clamp(0, 255).byte().cpu().numpy()


def make_model(cfg):
  return JEPA(cfg)


def make_policy(cfg): ## for BackgroundSimEval, which builds its own copy of the policy
    return FMPolicy(make_model(cfg), cfg)

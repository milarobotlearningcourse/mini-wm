## mini-FM: a small foundation model, built from micro-vlm (github.com/milarobotlearningcourse/micro-vlm): one
## transformer that predicts robot action chunks from a goal text, camera images and (optionally) the robot's pose. Its
## token head, from micro-vlm's VQA, gets no training rows. This file holds the models (FM, and SmallConvNetFM to compare
## with) and their sim-eval policy, FMPolicy; mini_train.py trains them, the data loading is in data_loading.py and the
## LIBERO sim evals in sim_eval.py.
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.profiler
from sim_eval import action_stats
from data_loading import scale_image, n_cameras

def plain_config(cfg):
    """ The config as nested types.SimpleNamespace objects. torch.compile traces attribute lookups on these, but not on
    OmegaConf's DictConfig, where each lookup splits the compiled graph in two. """
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    def convert(c):
        if isinstance(c, dict):
            return SimpleNamespace(**{k: convert(v) for k, v in c.items()})
        return [convert(v) for v in c] if isinstance(c, list) else c
    return convert(OmegaConf.to_container(cfg, resolve=True))


def n_action_tokens(cfg):
    """ The class tokens that end FM's sequence: one, whose features decode the whole action chunk, or with
    model.token_per_action one per action of the chunk (policy.action_stacking), each decoded by the same layer. """
    return cfg.policy.action_stacking if cfg.model.token_per_action else 1


def get_patches_fast(images, cfg):
    from einops import rearrange
    batch_size, height, width, channels = images.shape
    patch_size = cfg.patch_size ## n_patches = 8

    patches = rearrange(images[:,:,:,:3], 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
    if channels > 3:
        ## History stacking in the channel dimension for observations only, not goal images.
        patches = rearrange(images, 'b (h p1) (w p2) (c hs) -> b (h w hs) (p1 p2 c)', p1 = patch_size, p2 = patch_size, hs=cfg.policy.obs_stacking) ## Stack the history in the channel dimension
    return patches

def calc_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

class MultiHeadAttentionFast(nn.Module):
    """ Multi-head self-attention, all heads computed together in one batched operation """

    def __init__(self, num_heads, head_size, n_embd, dropout, cfg):
        super().__init__()
        self.num_heads, self.head_size = num_heads, head_size
        self.qkv = nn.Linear(n_embd, 3 * num_heads * head_size, bias=False) ## query, key and value for all heads at once
        ## The pose token (with policy.use_pose_data) and the image tokens start the sequence and attend to everything
        self.prefix_size = int(cfg.policy.use_pose_data) + cfg.image_block_size
        self.n_action_tokens = n_action_tokens(cfg)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def keys_values(self, x):
        """ The keys and values of every head, (2, B, n_head, T, head_size): what the diffusion head's action expert
        attends to at this layer. """
        B, T, _ = x.shape
        return self.qkv(x).view(B, T, 3, self.num_heads, self.head_size).permute(2, 0, 3, 1, 4)[1:]

    def forward(self, x, mask=None, key_mask=None):
        B, T, C = x.shape
        with torch.profiler.record_function("Self-Attention"):
            q, k, v = self.qkv(x).view(B, T, 3, self.num_heads, self.head_size).permute(2, 0, 3, 1, 4) # each (B, nh, T, hs)
            if mask is None: ### Block masked attention for the first pose and image tokens, then causal attention
                i = torch.arange(T, device=x.device)
                in_prefix = i < self.prefix_size
                mask = (i[None] <= i[:, None]) | in_prefix[:, None] ## (T, T), True where token [row] may attend to [column]
                if self.n_action_tokens > 1: ## the pose and image tokens attend only to each other, so the text and action
                    ## tokens are strictly causal: no later token reaches an earlier one through them a layer later
                    mask = mask & ~(in_prefix[:, None] & ~in_prefix[None])
            ## else (T, T), True where a token may attend: mini_wam.py's, for its longer sequence
            if key_mask is not None: ## (B, T), False for a sample's tokens no token may attend to (FM's missing cameras)
                mask = mask & key_mask[:, None, None, :]
            ## One fused kernel for the scores, mask, softmax, dropout and weighted sum, instead of each as its own pass
            ## over the (B, nh, T, T) scores. The scale is 1/sqrt(n_embd), as micro-vlm's, not SDPA's 1/sqrt(head_size).
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=C**-0.5,
                                                 dropout_p=self.attn_dropout.p if self.training else 0.0)
            out = out.transpose(1, 2).reshape(B, T, C) # re-assemble all head outputs side by side
            out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), ## This is where the information may be stored.
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x,)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, dropout, cfg):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttentionFast(n_head, head_size, n_embd=n_embd, dropout=dropout, cfg=cfg)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None, modulation=None, key_mask=None):
        ## modulation: mini_wam.py's adaLN-Zero (DiT, Peebles & Xie 2023), a shift, scale and gate per token for the
        ## attention and the feed-forward. Without it (as in mini-FM) they are 0, 0 and 1: the plain block.
        shift1, scale1, gate1, shift2, scale2, gate2 = modulation if modulation is not None else (0, 0, 1, 0, 0, 1)
        x = x + gate1 * self.sa(self.ln1(x) * (1 + scale1) + shift1, mask, key_mask)
        x = x + gate2 * self.ffwd(self.ln2(x) * (1 + scale2) + shift2)
        return x

class MSEActionHead(nn.Module):
  """ Regresses the (whitened) action chunk from the model's features, through net (a linear layer), with loss_name's
  loss: mse; l1, which OpenVLA-OFT found better for action chunks (MSE pulls a chunk with several likely modes to their
  mean); or huber, squared within 1 of the target (0.5 error^2) and L1 beyond (|error| - 0.5), so large errors pull no
  harder than in L1. """
  def __init__(self, net, loss_name="mse"):
    super().__init__()
    self.net = net
    self.loss_name = loss_name

  def loss(self, cond, actions): ## one loss per row
    predicted = self.net(cond).float()
    if self.loss_name == "huber":
      return F.huber_loss(predicted, actions, reduction="none", delta=1.0).mean(-1)
    error = predicted - actions
    return (error.abs() if self.loss_name == "l1" else error ** 2).mean(-1)

  def sample(self, cond):
    return self.net(cond)

  def log_prob(self, cond, actions, std):
    """ The log-density of each action value (every dimension of every step of the chunk) under a Gaussian around
    the predicted chunk with a fixed std, up to a constant: its squared error scaled by -1 / (2 std^2). GRPO
    fine-tuning (mini_fm_rl.py) uses it as the policy. """
    return -(self.net(cond).float() - actions) ** 2 / (2 * std ** 2)


def cosine_noise_schedule(steps):
  """ The noise added at each of a DDPM's steps (betas) and the fraction of the signal left after it (alpha_bar), with
  the cosine schedule of Nichol & Dhariwal 2021. """
  f = torch.cos((torch.linspace(0, 1, steps + 1) + 0.008) / 1.008 * math.pi / 2) ** 2
  betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
  return betas, torch.cumprod(1 - betas, dim=0)


class ActionExpertBlock(nn.Module):
  """ One layer of π0's action expert: a transformer block with its own weights, over the action tokens only. They
  attend to each other and to the backbone's keys and values at the same layer (the prefix); the backbone never
  attends to them. """

  def __init__(self, n_embd, n_head, dropout):
    super().__init__()
    self.n_head, self.dropout = n_head, dropout
    self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
    self.proj = nn.Linear(n_embd, n_embd)
    self.ffwd = FeedFoward(n_embd, dropout)

  def forward(self, x, prefix, prefix_mask):
    B, T, C = x.shape
    q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
    k, v = torch.cat((prefix[0], k), dim=2), torch.cat((prefix[1], v), dim=2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=prefix_mask, dropout_p=self.dropout if self.training else 0.0)
    x = x + self.proj(out.transpose(1, 2).reshape(B, T, C))
    return x + self.ffwd(self.ln2(x))


class DiffusionActionHead(nn.Module):
  """ Flow matching over the (whitened) action chunk with an action expert, as π0 (Black et al. 2024). Each action of
  the chunk is its own token: an MLP of its noisy action and the noise time τ, plus the embedding of its place in the
  chunk. The expert has one ActionExpertBlock per layer of the backbone, which reads that layer's keys and values (cond:
  the prefix, and a mask of the prefix tokens it may attend to, or None), and each token predicts the velocity from its
  clean action to its noise. Training mixes the chunk with noise at a random τ; acting starts from pure noise (τ = 1)
  and takes diffusion_steps Euler steps to τ = 0, reusing the prefix. Unlike the regression heads, it can model a chunk
  with several modes (e.g. going left or right around an object) instead of averaging them. """
  loss_name = "diffusion"

  def __init__(self, cfg):
    super().__init__()
    self.steps = cfg.model.diffusion_steps
    self.chunk, self.action_dim = cfg.policy.action_stacking, cfg.action_dim
    ## with FM's model.token_per_action, action i sees only the action tokens up to i, here and in the backbone, as the
    ## regression heads' action tokens do; otherwise the action tokens all see each other, as π0's
    self.causal = cfg.model.get("token_per_action", False)
    self.action_in = nn.Linear(cfg.action_dim, cfg.n_embd)
    self.action_time_mlp = nn.Sequential(nn.Linear(2 * cfg.n_embd, cfg.n_embd), nn.SiLU(), nn.Linear(cfg.n_embd, cfg.n_embd))
    self.chunk_positions = nn.Parameter(0.02 * torch.randn(self.chunk, cfg.n_embd))
    self.blocks = nn.ModuleList([ActionExpertBlock(cfg.n_embd, cfg.n_head, cfg.dropout) for _ in range(cfg.n_blocks)])
    self.velocity_out = nn.Sequential(nn.LayerNorm(cfg.n_embd), nn.Linear(cfg.n_embd, cfg.action_dim))

  def velocity(self, noisy_actions, time, cond):
    """ noisy_actions (B, chunk, action_dim) at noise times time (B,), cond (the prefix's keys and values at each layer,
    (layers, 2, B, heads, prefix tokens, head size), and its mask (B, prefix tokens) or None) -> the predicted velocity
    (B, chunk, action_dim). """
    prefix, prefix_mask = cond
    if prefix_mask is not None: ## (B, 1, 1, prefix + chunk): every action token is visible
        prefix_mask = torch.cat((prefix_mask, prefix_mask.new_ones(len(time), self.chunk)), dim=1)[:, None, None]
    if self.causal: ## (chunk, prefix + chunk): the backbone's action tokens end the prefix
        n_prefix = prefix.shape[-2]
        step = torch.arange(self.chunk, device=time.device)
        later = step[None] > step[:, None] ## [i, j]: action token j comes after action token i
        hidden = torch.cat((later.new_zeros(self.chunk, n_prefix - self.chunk), later, later), dim=1)
        prefix_mask = ~hidden if prefix_mask is None else prefix_mask & ~hidden
    actions = self.action_in(noisy_actions)
    half = actions.shape[-1] // 2 ## sinusoidal features of τ, as the positions of a transformer
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=time.device) / half)
    time_features = torch.cat((torch.sin(1000 * time[:, None] * frequencies), torch.cos(1000 * time[:, None] * frequencies)), dim=-1)
    x = self.action_time_mlp(torch.cat((actions, time_features[:, None].expand_as(actions)), dim=-1)) + self.chunk_positions
    for block, layer_prefix in zip(self.blocks, prefix):
        x = block(x, layer_prefix, prefix_mask)
    return self.velocity_out(x).float()

  def loss(self, cond, actions):
    actions = actions.view(len(actions), self.chunk, self.action_dim) ## the chunk is flattened step by step
    ## τ ~ Beta(1.5, 1) (sampled as u^(1/1.5)), so more training goes to the noisier times, as π0
    time = 0.001 + 0.999 * torch.rand(len(actions), device=actions.device) ** (2 / 3)
    noise = torch.randn_like(actions)
    noisy_actions = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    return ((self.velocity(noisy_actions, time, cond) - (noise - actions)) ** 2).mean((1, 2)) ## one loss per row

  @torch.no_grad()
  def sample(self, cond):
    prefix = cond[0]
    x = torch.randn(prefix.shape[2], self.chunk, self.action_dim, device=prefix.device)
    for i in range(self.steps): ## Euler steps from τ = 1 (noise) to 0 (the chunk)
        time = torch.full((len(x),), 1 - i / self.steps, device=x.device)
        x = x - self.velocity(x, time, cond) / self.steps
    return x.flatten(1)


class FM(nn.Module):
  def __init__(self, cfg, mlp_ratio=4):
    super(FM, self).__init__()
    self._cfg = cfg
    self._plain_cfg = plain_config(cfg) ## what forward() reads, so the compiled forward is one graph
    ## image_block_size is the number of image tokens: 0 trains on text only, otherwise one token per image patch
    n_patches = int(cfg.image_shape[0] / cfg.patch_size) * int(cfg.image_shape[1] / cfg.patch_size) * n_cameras(cfg) ## cameras side by side
    self.n_cameras = n_cameras(cfg)
    if self.n_cameras > 1: ## the camera of each image token (get_patches_fast's order: row by row, across all cameras)
        width = int(cfg.image_shape[1] / cfg.patch_size) ## patches across one camera
        self.register_buffer("token_camera", torch.arange(n_patches) % (width * self.n_cameras) // width, persistent=False)
    assert cfg.image_block_size in (0, n_patches), f"image_block_size must be 0 or {n_patches} (one token per patch), got {cfg.image_block_size}"
    #Positional embedding
    self.register_buffer('positional_embeddings', calc_positional_embeddings(int(cfg.policy.use_pose_data) + ## pose
                      cfg.image_block_size + ## image
                      self._cfg.max_block_size + n_action_tokens(cfg), ## text and class token(s)
                      cfg.n_embd), 
                      persistent=False)

    self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
    self.n_action_tokens = n_action_tokens(cfg)
    self.class_tokens = nn.Parameter(torch.rand(self.n_action_tokens, cfg.n_embd))

    self.input_d = int(self._cfg.image_shape[2] * cfg.patch_size * cfg.patch_size)

    self.lin_map = nn.Linear(self.input_d, self._cfg.n_embd, bias=False) 
    if cfg.policy.use_pose_data: ## one pose token before the image tokens; no bias, so non-robot rows (zero pose) get a zero token, like a blank image's patches
        self.lin_map_pose = nn.Linear(cfg.pose_dim, cfg.n_embd, bias=False)

    # 4) Transformer encoder blocks
    self.blocks = nn.ModuleList([Block(self._cfg.n_embd, self._cfg.n_head, dropout=self._cfg.dropout, cfg=cfg) for _ in range(self._cfg.n_blocks)])

    # 5) Classification MLPk
    self.mlp = nn.Sequential(
        nn.Linear(self._cfg.n_embd, self._cfg.action_bins),  # Output size is action_bins 
    )
    ## Continuous action chunks for robot rows: policy.action_stacking actions of action_dim
    if cfg.model.action_head == "diffusion": ## its action expert attends to every token of every layer
        self.action_head = DiffusionActionHead(cfg)
    elif cfg.model.token_per_action: ## each action token's features -> its action, then the chunk flattened step by step
        self.action_head = MSEActionHead(nn.Sequential(nn.Linear(cfg.n_embd, cfg.action_dim), nn.Flatten(1)), cfg.model.action_head)
    else:
        self.action_head = MSEActionHead(nn.Linear(cfg.n_embd, cfg.action_dim * cfg.policy.action_stacking), cfg.model.action_head)

  def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
          if module.bias is not None:
              torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, X_text, X_images, targets=None, actions=None, pose=None):
    cfg = self._plain_cfg
    n = X_images.shape[0]
    context = self.token_embedding_table(X_text) ## one token per character of the goal text

    # Running linear layer tokenization to get embeddings
    # Map the vector corresponding to each patch to the hidden size dimension
    if cfg.image_block_size == 0: ## Text only: no image tokens in the sequence
        out_obs = context.new_zeros(n, 0, cfg.n_embd)
    else:
        ## Text-only rows use a blank image; their patch embeddings are zeroed. A multiply instead of selecting the rows
        ## with images keeps the shapes fixed, so torch.compile captures the forward as one graph.
        has_img = (X_images != cfg.image_bounds[0]).flatten(1).any(dim=1) ## a blank image is all low bound
        out_obs = self.lin_map(get_patches_fast(X_images, cfg)) * has_img[:, None, None]
    
    if cfg.policy.use_pose_data:
        out_obs = torch.cat((self.lin_map_pose(pose)[:, None], out_obs), dim=1)
    out = torch.cat((out_obs, context, self.class_tokens.expand(n, -1, -1)), dim=1)
    
    # Adding positional embedding
    out = out + self.positional_embeddings.repeat(n, 1, 1)
    
    ## With several cameras, the tokens of a camera a sample's dataset doesn't have (a blank image there: all low bound)
    ## are hidden from attention, so the model never reads them
    key_mask = None
    if self.n_cameras > 1:
        cameras = X_images.unflatten(2, (self.n_cameras, -1)) ## (B, H, cameras, W, C)
        has_camera = (cameras != cfg.image_bounds[0]).any(dim=4).any(dim=3).any(dim=1) ## (B, cameras)
        key_mask = torch.cat((has_camera.new_ones(n, int(cfg.policy.use_pose_data)), has_camera[:, self.token_camera],
                              has_camera.new_ones(n, out.shape[1] - out_obs.shape[1])), dim=1)

    # Transformer Blocks
    diffusion = cfg.model.action_head == "diffusion"
    prefix = [] ## with the diffusion head, each layer's keys and values, which its action expert attends to
    for block in self.blocks:
        if diffusion:
            prefix.append(block.sa.keys_values(block.ln1(out)))
        out = block(out, key_mask=key_mask)

    # Getting the class token(s) only: (B, n_embd), or (B, action_stacking, n_embd) with model.token_per_action
    out = out[:, -1] if self.n_action_tokens == 1 else out[:, -self.n_action_tokens:]
    cond = (torch.stack(prefix), key_mask) if diffusion else out ## what the action head reads
    if targets is False: ## from act()
        return cond
    logits = self.mlp(out if self.n_action_tokens == 1 else out[:, -1]) ## the token head reads the last class token
        
    parts = {} ## The loss of each head, for logging
    if targets is None:
        loss = None
    else:
        ## Robot rows (target -1) use the action head; all other rows use the token head and cross entropy. Each head's
        ## loss is its mean over its own rows, computed for every row and masked, so there are no branches on the data;
        ## a head with no rows in the batch gets 0.
        is_robot = (targets < 0).float()
        ce = F.cross_entropy(logits.float(), targets.clamp(min=0), reduction="none")
        parts = {"ce": (ce * (1 - is_robot)).sum() / (1 - is_robot).sum().clamp(min=1),
                 self.action_head.loss_name: (self.action_head.loss(cond, actions) * is_robot).sum() / is_robot.sum().clamp(min=1)}
        ## parts stay unweighted for logging; mse_head_weights scales the action head so it isn't swamped by the larger CE.
        loss = parts["ce"] + cfg.mse_head_weights * parts[self.action_head.loss_name]
    return (logits, loss, parts)
  
  def act(self, X_text, X_images, pose):
    """ The (whitened) action chunk predicted for each row, flattened. """
    return self.action_head.sample(self(X_text, X_images, targets=False, pose=pose))

  def generate(self, idx, image, pose, max_new_tokens, cfg):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -cfg.max_block_size:]
            # get the predictions
            logits, loss, _ = self(idx_cond, image, None, pose=pose)
            # focus only on the last time step
            logits = logits # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


class SmallConvNetFM(nn.Module):
  """ A very small policy behind FM's interface: three stride-2 convolutions of the camera image (one camera), the goal
  text's character embeddings, one hidden layer and a regression action head (model.action_head: huber, mse or l1).
  Cheap enough to act at every step of mini-Dreamer's imagination (model.policy_net: small_convnet). """

  def __init__(self, cfg):
    super().__init__()
    assert cfg.model.action_head != "diffusion", "small_convnet has a regression action head only"
    self._cfg = cfg
    H, W, C = cfg.image_shape
    self.encoder = nn.Sequential( ## 64x64 -> 8x8
        nn.Conv2d(C, 32, 4, stride=2, padding=1), nn.SiLU(),
        nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.SiLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1), nn.SiLU(),
        nn.Flatten())
    self.text_embedding = nn.Embedding(cfg.vocab_size, 16) ## flattened, so the characters' order is kept
    hidden = 256
    self.trunk = nn.Sequential(
        nn.Linear(64 * (H // 8) * (W // 8) + 16 * cfg.max_block_size, hidden), nn.LayerNorm(hidden), nn.SiLU())
    self.action_head = MSEActionHead(nn.Linear(hidden, cfg.action_dim * cfg.policy.action_stacking), cfg.model.action_head)

  def forward(self, X_text, X_images, targets=None, actions=None, pose=None):
    cond = self.trunk(torch.cat((self.encoder(X_images.permute(0, 3, 1, 2)), self.text_embedding(X_text).flatten(1)), -1))
    if targets is False: ## from act()
      return cond
    parts = {} if targets is None else {self.action_head.loss_name: self.action_head.loss(cond, actions).mean()}
    loss = self._cfg.mse_head_weights * parts[self.action_head.loss_name] if parts else None ## weighted as FM's action head
    return (None, loss, parts)

  def act(self, X_text, X_images, pose):
    return self.action_head.sample(self(X_text, X_images, targets=False, pose=pose))


POLICY_NETS = {"fm": FM, "small_convnet": SmallConvNetFM}

def make_model(cfg):
  return POLICY_NETS[cfg.model.type](cfg)


class FMPolicy(nn.Module):
    """ Wraps a model (FM, SmallConvNetFM, or a world model's policy) in the interface sim_eval.py's rollouts expect:
    it encodes the goal text with dataset.chars_list and whitens the pose with dataset.pose_mean/std. It un-whitens the
    action chunk with the stats of the dataset of the task set it runs in (task_set, set by the sim eval or RL). The sim
    renders the camera the model was trained on (sim.obs_key, set by the eval from sim_eval.image_key_of) at the training
    image size; rollout_libero flips it vertically, and the dataset stored it rotated 180 degrees, so a horizontal flip
    maps it to the training images. """

    def __init__(self, model, cfg):
        super().__init__()
        self._cfg = cfg
        self.model = model
        self.task_set = None ## e.g. libero_90

    def encode_text_goal(self, goal): ## the first max_block_size characters, padded with spaces, as the batches
        stoi = {c: i for i, c in enumerate(self._cfg.dataset.chars_list)}
        ids = [stoi.get(c, stoi[" "]) for c in goal[:self._cfg.max_block_size].ljust(self._cfg.max_block_size)]
        return torch.tensor([ids], dtype=torch.long, device=self._cfg.device)

    def encode_pose(self, pose):
        mean, std = (torch.tensor(x, dtype=pose.dtype, device=pose.device) for x in (self._cfg.dataset.pose_mean, self._cfg.dataset.pose_std))
        return (pose - mean) / std

    def decode_action(self, action): ## an action chunk, (..., action_stacking * action_dim)
        mean, std = (torch.tensor(x, dtype=action.dtype, device=action.device).repeat(self._cfg.policy.action_stacking)
                     for x in action_stats(self._cfg, self.task_set))
        return action * std + mean

    def preprocess_state(self, obs): ## scaled as in to_device()
        return scale_image(np.ascontiguousarray(obs[:, ::-1, :3]).astype(np.float32), self._cfg.image_bounds)

    def robot_pose(self, obs): ## the arm's joint positions, as the datasets' POSE_KEY
        return obs["robot0_joint_pos"]

    def forward(self, obs, txt_goal, goal_img, pose=None): ## a batch of observations
        n = len(obs)
        pose = pose.reshape(n, -1).float() if pose is not None else None
        if getattr(self.model, "goal_conditioned", False): ## a planner (mini_jepa.py's CEM): towards the episode's goal image, without the goal text
            return self.model.plan_to(obs.float(), goal_img.float()).float(), None
        return self.model.act(txt_goal.reshape(n, -1).long(), obs.float(), pose).float(), None ## fp32 under autocast, for decode_action


def make_policy(cfg): ## for BackgroundSimEval, which builds its own copy of the policy
    return FMPolicy(make_model(cfg), cfg)

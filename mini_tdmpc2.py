## mini-TD-MPC2: TD-MPC2 (Hansen et al. 2024, github.com/nicklashansen/tdmpc2) in one file, at its 5M size, learned
## offline from robot demonstrations. A world model with no decoder: a CNN encodes the camera frame into a latent z
## (SimNorm: groups of 8 softmaxed), and MLPs predict, from z and an action, the next z (dynamics), the reward and the
## discounted return (an ensemble of Q-functions), and the action itself (the policy prior pi). It learns, over a horizon
## of model.horizon steps from each sampled frame:
## - the dynamics by latent consistency: each predicted z close (MSE) to the encoding of the real next frame;
## - the rewards and Q-values by cross-entropy on a two-hot encoding of symlog(target) over 101 bins, Q towards the TD
##   target r + discount * Q_target(z', pi(z')), from the minimum of two random Q-functions of a slowly moving copy;
## - the policy prior pi by maximizing Q (with a little entropy), on the predicted latents.
## To act it plans with MPPI: it samples action sequences (some from pi, the rest from a Gaussian it refits to the best
## ones, model.iterations times), scores each by the predicted rewards over the horizon plus Q at its end, and takes the
## first action of a good one (model.mpc false: pi's mean instead).
## Adapted to this code's data: one step is an action chunk (policy.action_stacking actions, which the sim eval runs
## before asking again), in whitened units scaled by model.action_limit into [-1, 1]; the reward is 1 for a step after
## which the demonstration has ended (it reached its goal; RobotCache repeats the last frame past the end, so that state
## keeps the reward); the task embedding is a learned linear map of the goal text (TD-MPC2's multi-task embedding, but
## read from the text, so a sim task with the same goal gets the same one). Left out of TD-MPC2: its separate optimizers
## (and their gradient clipping and slower encoder), and the planner's warm start from the previous step's plan.
## mini_train.py trains it (--config-name mini-tdmpc2).
import copy
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from mini_fm import plain_config, FMPolicy


class SimNorm(nn.Module):
  """ Simplicial normalization: a softmax over each group of 8 features, so the latent stays bounded. """
  def forward(self, x):
    return x.unflatten(-1, (-1, 8)).softmax(-1).flatten(-2)


def mlp(in_dim, hidden_dim, out_dim, out_act=None, dropout=0.0):
  """ TD-MPC2's MLP: two hidden layers, each Linear, LayerNorm and Mish (dropout after the first Linear), then a
  Linear, or with out_act a Linear, LayerNorm and out_act. """
  layer = lambda i, o, act, p=0.0: [nn.Linear(i, o), *([nn.Dropout(p)] if p else []), nn.LayerNorm(o), act]
  out = layer(hidden_dim, out_dim, out_act) if out_act else [nn.Linear(hidden_dim, out_dim)]
  return nn.Sequential(*layer(in_dim, hidden_dim, nn.Mish(), dropout), *layer(hidden_dim, hidden_dim, nn.Mish()), *out)


class Ensemble(nn.ModuleList):
  """ The Q-functions: their outputs stacked, (num_q, ..., num_bins). """
  def forward(self, x):
    return torch.stack([q(x) for q in self])


def symlog(x):
  return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
  return torch.sign(x) * torch.expm1(x.abs())


class TDMPC2(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self._cfg = cfg
    self._plain_cfg = plain_config(cfg)
    m = cfg.model
    assert tuple(cfg.image_shape[:2]) == (64, 64), "TD-MPC2's CNN takes 64x64 frames (one camera)"
    self.chunk, self.action_dim = cfg.policy.action_stacking, cfg.policy.action_stacking * cfg.action_dim ## one step is an action chunk
    latent, hidden = m.latent_dim, m.mlp_dim
    ## The CNN: 64 -> 29 -> 13 -> 6 -> 4 pixels, 32 channels, flattened to the latent (512)
    self.encoder = nn.Sequential(
        nn.Conv2d(3, 32, 7, stride=2), nn.ReLU(), nn.Conv2d(32, 32, 5, stride=2), nn.ReLU(),
        nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1), nn.Flatten(), SimNorm())
    assert 32 * 4 * 4 == latent, "the CNN's output is the latent: model.latent_dim 512"
    self.vocab_size = cfg.vocab_size
    self.task_embedding = nn.Linear(cfg.max_block_size * cfg.vocab_size, m.task_dim, bias=False) ## one vector per character and position
    self.dynamics = mlp(latent + self.action_dim + m.task_dim, hidden, latent, out_act=SimNorm())
    self.reward_head = mlp(latent + self.action_dim + m.task_dim, hidden, m.num_bins)
    self.policy = mlp(latent + m.task_dim, hidden, 2 * self.action_dim)
    self.Qs = Ensemble([mlp(latent + self.action_dim + m.task_dim, hidden, m.num_bins, dropout=m.dropout) for _ in range(m.num_q)])
    self.apply(self.weight_init)
    for head in (self.reward_head, *self.Qs): ## predict 0 at first
        nn.init.zeros_(head[-1].weight)
    self.target_Qs = copy.deepcopy(self.Qs).requires_grad_(False) ## the moving average of the Q-functions, for the TD targets
    self.register_buffer("bins", torch.linspace(m.vmin, m.vmax, m.num_bins), persistent=False) ## symlog values of the bins
    self.register_buffer("q_scale", torch.ones(())) ## running scale of the Q-values in the policy loss (TD-MPC2's RunningScale)

  @staticmethod
  def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

  def train(self, mode=True):
    super().train(mode)
    self.target_Qs.train(False) ## no dropout in the targets
    return self

  @torch.no_grad()
  def update_target(self):
    """ Moves the target Q-functions towards the Q-functions by model.tau. Called by mini_train.py after each optimizer
    step. """
    torch._foreach_lerp_(list(self.target_Qs.parameters()), list(self.Qs.parameters()), self._plain_cfg.model.tau)

  ## The two-hot encoding of the rewards and returns: a value is symlog'd and split between its two nearest bins
  def two_hot(self, x):
    x = symlog(x).clamp(self.bins[0], self.bins[-1])
    return F.relu(1 - (x[..., None] - self.bins).abs() / (self.bins[1] - self.bins[0]))

  def scalar(self, logits):
    """ The value of a reward's or Q-function's output: the expected bin, symexp'd. """
    return symexp((logits.float().softmax(-1) * self.bins).sum(-1))

  def soft_ce(self, logits, x):
    return -(self.two_hot(x) * logits.float().log_softmax(-1)).sum(-1)

  def task(self, X_text):
    """ The task embedding of each goal text, (B, task_dim), scaled to at most norm 1 (as nn.Embedding's max_norm). """
    emb = self.task_embedding(F.one_hot(X_text, self.vocab_size).flatten(1).float())
    return emb / emb.norm(dim=-1, keepdim=True).clamp(min=1)

  def encode(self, images):
    """ Images (..., H, W, 3), pixels in image_bounds, to their latents (..., latent_dim). """
    low, high = self._plain_cfg.image_bounds
    x = ((images - low) / (high - low) - 0.5).flatten(0, -4).permute(0, 3, 1, 2) ## in [-0.5, 0.5], as TD-MPC2
    return self.encoder(x).unflatten(0, images.shape[:-3])

  def next(self, z, a, task):
    return self.dynamics(torch.cat((z, a, task), -1))

  def reward(self, z, a, task):
    return self.reward_head(torch.cat((z, a, task), -1))

  def pi(self, z, task):
    """ A sample of the policy prior, its mean (both squashed into [-1, 1] by tanh) and its scaled entropy, as
    TD-MPC2's: the Gaussian's log-probability times the action's dimensions (so the entropy bonus doesn't shrink as
    the action grows). """
    m = self._plain_cfg.model
    mean, log_std = self.policy(torch.cat((z, task), -1)).float().chunk(2, -1)
    log_std = m.log_std_min + 0.5 * (m.log_std_max - m.log_std_min) * (torch.tanh(log_std) + 1)
    eps = torch.randn_like(mean)
    log_prob = (-0.5 * eps ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
    scaled_log_prob = log_prob * eps.shape[-1]
    action = torch.tanh(mean + eps * log_std.exp())
    squashed_log_prob = log_prob - torch.log(F.relu(1 - action ** 2) + 1e-6).sum(-1)
    return action, torch.tanh(mean), -squashed_log_prob * scaled_log_prob / (squashed_log_prob + 1e-8)

  def Q(self, z, a, task, how, Qs=None):
    """ Q-values from two of the Q-functions picked at random: their minimum (how "min") or mean ("avg"). Qs: the
    ensemble to use (the target one, or one with detached weights), the trained one by default. """
    logits = (self.Qs if Qs is None else Qs)(torch.cat((z, a, task), -1))
    two = self.scalar(logits[torch.randperm(len(logits))[:2]])
    return two.min(0).values if how == "min" else two.mean(0)

  def forward(self, X_text, X_images, targets=None, actions=None, pose=None, frames=None, sequence_actions=None, rewards=None):
    """ The TD-MPC2 losses on the current frame (X_images) and the frames after each of the next horizon action
    chunks (frames: the frames after each action, stacked in the channels, as RobotCache's future frames;
    sequence_actions: those actions; rewards: 1 after each action from which the episode had ended). Returns
    (None, loss, parts). """
    loss, parts = self.loss(X_text, X_images, frames, sequence_actions, rewards) ## unpacked here: torch 2.5.1's compile fails on (None, *self.loss(...))
    return None, loss, parts

  ## Run eagerly: compiling it gains little for these small MLPs, and its random picks of Q-functions break the graph
  @torch.compiler.disable
  def loss(self, X_text, X_images, frames, sequence_actions, rewards):
    m = self._plain_cfg.model
    H, rho = m.horizon, m.rho ** torch.arange(m.horizon + 1, device=X_images.device)
    ## Each step is a chunk: the frame and reward after its last action, its actions scaled into [-1, 1]
    next_images = frames.unflatten(-1, (-1, 3))[..., self.chunk - 1::self.chunk, :].movedim(-2, 1) ## (B, H, h, w, 3)
    actions = (sequence_actions.unflatten(1, (H, -1)) / m.action_limit).clamp(-1, 1) ## (B, H, chunk * action_dim)
    rewards = rewards[:, self.chunk - 1::self.chunk] ## (B, H)
    task = self.task(X_text)
    task_h = task[:, None].expand(-1, H, -1)
    with torch.no_grad():
        next_z = self.encode(next_images) ## (B, H, latent)
        td_targets = rewards + m.discount * self.Q(next_z, self.pi(next_z, task_h)[0], task_h, "min", self.target_Qs)
    ## Roll the dynamics from the current frame along the actions taken
    zs = [self.encode(X_images)]
    for t in range(H):
        zs.append(self.next(zs[-1], actions[:, t], task))
    zs = torch.stack(zs, 1) ## (B, H + 1, latent)
    consistency = ((zs[:, 1:] - next_z).float() ** 2).mean((0, 2))
    reward = self.soft_ce(self.reward(zs[:, :-1], actions, task_h), rewards).mean(0)
    value = self.soft_ce(self.Qs(torch.cat((zs[:, :-1], actions, task_h), -1)), td_targets).mean((0, 1))
    ## Each step's loss weighted by rho^t, then averaged over the horizon
    weighted = lambda per_step: (per_step * rho[:len(per_step)]).mean()
    world_model = m.consistency_coef * weighted(consistency) + m.reward_coef * weighted(reward) + m.value_coef * weighted(value)
    ## The policy prior: maximize Q (scaled by the running spread of Q) on the predicted latents, which, like the
    ## Q-functions and the task embedding, it doesn't train
    zs, task_pi = zs.detach(), task[:, None].expand(-1, H + 1, -1).detach()
    action, _, entropy = self.pi(zs, task_pi)
    detached_Qs = torch.func.functional_call(self.Qs, {n: p.detach() for n, p in self.Qs.named_parameters()},
                                             (torch.cat((zs, action, task_pi), -1),))
    q = self.scalar(detached_Qs[torch.randperm(m.num_q)[:2]]).mean(0) ## (B, H + 1)
    if self.training:
        spread = torch.quantile(q[:, 0].detach(), torch.tensor([0.05, 0.95], device=q.device))
        self.q_scale.lerp_((spread[1] - spread[0]).clamp(min=1), m.tau)
    pi_loss = weighted(-(m.entropy_coef * entropy + q / self.q_scale).mean(0))
    parts = {"consistency": consistency.mean(), "reward_ce": reward.mean(),
             "value_ce": value.mean(), "pi_loss": pi_loss, "q": q[:, 0].mean(), "reward_rate": rewards.mean()}
    return world_model + pi_loss, parts

  @torch.no_grad()
  def plan(self, z, task):
    """ MPPI: the first action chunk of a good action sequence from each latent z (B, latent), in [-1, 1]. """
    m = self._plain_cfg.model
    B, H, N, P, A = len(z), m.horizon, m.num_samples, m.num_pi_trajs, self.action_dim
    iterations = m.iterations + 2 * (A >= 20) ## TD-MPC2's heuristic for large action spaces
    ## P action sequences from the policy prior, through the dynamics
    z_pi, task_pi, pi_actions = z.repeat_interleave(P, 0), task.repeat_interleave(P, 0), []
    for t in range(H):
        pi_actions.append(self.pi(z_pi, task_pi)[0])
        z_pi = self.next(z_pi, pi_actions[-1], task_pi)
    pi_actions = torch.stack(pi_actions, 1).unflatten(0, (B, P)) ## (B, P, H, A)
    z, task = z.repeat_interleave(N, 0), task.repeat_interleave(N, 0)
    mean, std = torch.zeros(B, H, A, device=z.device), torch.full((B, H, A), m.max_std, device=z.device)
    for _ in range(iterations):
        sampled = (mean[:, None] + std[:, None] * torch.randn(B, N - P, H, A, device=z.device)).clamp(-1, 1)
        actions = torch.cat((pi_actions, sampled), 1) ## (B, N, H, A)
        value = self.estimate_value(z, actions.flatten(0, 1), task).unflatten(0, (B, N)).nan_to_num(0)
        elite_value, elite = value.topk(m.num_elites, dim=1)
        elite_actions = actions[torch.arange(B, device=z.device)[:, None], elite].float() ## (B, elites, H, A)
        score = (m.temperature * (elite_value - elite_value.max(1, keepdim=True).values)).exp()
        score = (score / score.sum(1, keepdim=True))[..., None, None]
        mean = (score * elite_actions).sum(1)
        std = (score * (elite_actions - mean[:, None]) ** 2).sum(1).sqrt().clamp(m.min_std, m.max_std)
    pick = torch.multinomial(score[..., 0, 0], 1)[:, 0] ## an elite, by its score
    return elite_actions[torch.arange(B, device=z.device), pick, 0]

  def estimate_value(self, z, actions, task):
    """ The discounted predicted rewards along each action sequence (N, H, A), plus the discounted Q at its end. """
    G, discount = 0, 1
    for t in range(actions.shape[1]):
        G = G + discount * self.scalar(self.reward(z, actions[:, t], task))
        z = self.next(z, actions[:, t], task)
        discount = discount * self._plain_cfg.model.discount
    return G + discount * self.Q(z, self.pi(z, task)[0], task, "avg")

  @torch.no_grad()
  def act(self, X_text, X_images, pose):
    """ The (whitened) action chunk for each row, flattened: planned (model.mpc), or the policy prior's mean. """
    z, task = self.encode(X_images), self.task(X_text)
    action = self.plan(z, task) if self._plain_cfg.model.mpc else self.pi(z, task)[1]
    return action * self._plain_cfg.model.action_limit


def make_model(cfg):
  return TDMPC2(cfg)


def make_policy(cfg): ## for BackgroundSimEval, which builds its own copy of the policy
    return FMPolicy(make_model(cfg), cfg)

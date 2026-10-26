## mini-Dreamer: a small forward dynamics model p(s'|s, a), the world model of DreamerV3 (Hafner et al. 2023), on robot
## data: from the current camera frame and the next model.sequence_steps actions taken,
## it predicts the frame after each action, at model.frame_size pixels. A CNN encodes each frame; the RSSM's GRU steps
## a deterministic state h with the last latent z and the action; the prior predicts z from h alone, the posterior from h
## and the frame's encoding; each frame is decoded from (h, z). In training z comes from the posterior; the KL between the
## two teaches the prior to predict it. model.latent picks z: categorical (DreamerV3: stoch_dim one-hot variables of
## discrete_dim classes) or gaussian (DreamerV1: a stoch_dim Gaussian). The world model is its own network: mini-FM
## (mini_fm.py; or a small convnet, model.policy_net), trained next to it, is the policy that the sim evals run: Dreamer's actor. Besides imitating the
## demonstrations, it learns in imagination: from each real frame's state it acts on the decoded frames for
## model.imagine_horizon steps in the world model, which predicts the rewards and the critic's value at the end, and
## mini-FM is updated by REINFORCE on those imagined trajectories, as the actor loss of DreamerV2 (arXiv 2010.02193,
## "Actor loss"; no gradients through the world model). Its
## reward head predicts, from (h, z), the reward of TD-MPC2's (1 after an action from which the demonstration has
## reached its last frame); its critic learns the value of each state of the demonstrations: the lambda-return of their
## rewards, bootstrapped with a slow-moving copy of the critic (DreamerV3's, on the real sequence, not imagined ones).
## Both train the RSSM too, as the decoder does, so (h, z) holds what predicts rewards and values.
## mini_train.py trains it (--config-name mini-dreamer).
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from mini_fm import plain_config, POLICY_NETS, FMPolicy


def symlog(x):
  return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
  return torch.sign(x) * torch.expm1(x.abs())


class ChannelNorm(nn.LayerNorm):
  """ LayerNorm over the channels of each pixel of (B, C, H, W) images, as DreamerV3's CNNs. """
  def forward(self, x):
    return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Dreamer(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self._cfg = cfg
    self._plain_cfg = plain_config(cfg) ## what forward() reads, so the compiled forward is one graph
    self.fm = POLICY_NETS[cfg.model.policy_net](cfg) ## the policy: mini-FM's transformer, or a small convnet (model.policy_net)
    m = cfg.model
    self.n_steps, self.categorical = m.sequence_steps, m.latent == "categorical"
    ## Imagination steps the world model on each action mini-FM predicts, and REINFORCE needs its log-probability: a
    ## Gaussian around a regression head's prediction (the diffusion head has none)
    assert m.actor_coef == 0 or (cfg.policy.action_stacking == 1 and m.action_head != "diffusion"), \
        "imagination (model.actor_coef > 0) needs policy.action_stacking 1 and a regression action head"
    self.z_dim = m.stoch_dim * m.discrete_dim if self.categorical else m.stoch_dim
    z_params = self.z_dim if self.categorical else 2 * m.stoch_dim ## logits, or a mean and std
    ## The CNNs: 4 layers of stride 2, so the encoder's output is (256, frame_size / 16, frame_size / 16)
    self.enc_shape = (256, m.frame_size // 16, m.frame_size // 16)
    embed_dim = self.enc_shape[0] * self.enc_shape[1] * self.enc_shape[2]
    self.encoder = nn.Sequential(
        nn.Conv2d(3, 32, 4, stride=2, padding=1), ChannelNorm(32), nn.SiLU(),
        nn.Conv2d(32, 64, 4, stride=2, padding=1), ChannelNorm(64), nn.SiLU(),
        nn.Conv2d(64, 128, 4, stride=2, padding=1), ChannelNorm(128), nn.SiLU(),
        nn.Conv2d(128, 256, 4, stride=2, padding=1), ChannelNorm(256), nn.SiLU(),
        nn.Flatten())
    ## The RSSM
    self.gru = nn.GRUCell(self.z_dim + cfg.action_dim, m.deter_dim)
    self.prior_net = nn.Sequential(
        nn.Linear(m.deter_dim, m.hidden_dim), nn.LayerNorm(m.hidden_dim), nn.SiLU(), nn.Linear(m.hidden_dim, z_params))
    self.posterior_net = nn.Sequential(
        nn.Linear(m.deter_dim + embed_dim, m.hidden_dim), nn.LayerNorm(m.hidden_dim), nn.SiLU(), nn.Linear(m.hidden_dim, z_params))
    ## The decoder: the features (h, z) mapped to the encoder's output shape, then the encoder mirrored
    self.decoder_input_net = nn.Linear(m.deter_dim + self.z_dim, embed_dim)
    self.decoder_net = nn.Sequential(
        ChannelNorm(256), nn.SiLU(),
        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), ChannelNorm(128), nn.SiLU(),
        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), ChannelNorm(64), nn.SiLU(),
        nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), ChannelNorm(32), nn.SiLU(),
        nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1))
    ## The reward head and the critic, on the features (h, z), both in symlog space, as DreamerV3's
    head = lambda: nn.Sequential(
        nn.Linear(m.deter_dim + self.z_dim, m.hidden_dim), nn.LayerNorm(m.hidden_dim), nn.SiLU(), nn.Linear(m.hidden_dim, 1))
    self.reward_head, self.critic = head(), head()
    self.target_critic = copy.deepcopy(self.critic).requires_grad_(False) ## the slow critic, for the return targets

  def update_target(self):
    """ Moves the slow critic towards the critic by model.critic_tau. Called by mini_train.py after each optimizer
    step. """
    torch._foreach_lerp_(list(self.target_critic.parameters()), list(self.critic.parameters()), self._plain_cfg.model.critic_tau)

  def small_frames(self, images):
    """ Frames stacked in the channels, (B, H, W, n * 3) as the camera images or RobotCache's future frames,
    downsampled to frame_size: (B, n, 3, frame_size, frame_size). """
    small = F.avg_pool2d(images.permute(0, 3, 1, 2), images.shape[1] // self._plain_cfg.model.frame_size)
    return small.unflatten(1, (-1, 3))

  def categorical_log_probs(self, params):
    """ The log-probabilities of z's categorical variables, (..., stoch_dim, discrete_dim): the network's softmax mixed
    with 1% uniform, as DreamerV3's unimix, so no class's probability reaches 0 and the KL can't blow up. """
    probs = params.float().unflatten(-1, (self._plain_cfg.model.stoch_dim, -1)).softmax(-1)
    return (0.99 * probs + 0.01 / probs.shape[-1]).log()

  def sample_stochastic(self, params, training=True):
    """ A sample of z from the prior's or posterior's output. Categorical: one-hot per variable, by Gumbel-softmax
    with straight-through gradients in training, the most likely class otherwise. Gaussian: the mean plus noise in
    training, the mean otherwise. """
    if self.categorical:
        log_probs = self.categorical_log_probs(params)
        z = F.gumbel_softmax(log_probs, tau=1.0, hard=True, dim=-1) if training else F.one_hot(log_probs.argmax(-1), log_probs.shape[-1]).float()
        return z.flatten(-2).to(params.dtype)
    mean, std = self.gaussian(params)
    return mean + std * torch.randn_like(std) if training else mean

  def gaussian(self, params):
    """ The mean and std of a Gaussian z, the std kept above 0.1, as DreamerV1. """
    mean, std = params.float().chunk(2, dim=-1)
    return mean, F.softplus(std) + 0.1

  def kl(self, p, q):
    """ KL(p || q) of two of z's distributions, given by the networks' outputs, summed over z's variables: (B,). """
    if self.categorical:
        log_p, log_q = self.categorical_log_probs(p), self.categorical_log_probs(q)
        return (log_p.exp() * (log_p - log_q)).sum((-2, -1))
    (mean_p, std_p), (mean_q, std_q) = self.gaussian(p), self.gaussian(q)
    return (torch.log(std_q / std_p) + (std_p ** 2 + (mean_p - mean_q) ** 2) / (2 * std_q ** 2) - 0.5).sum(-1)

  def rssm_step(self, h, z, action, embed=None, training=True):
    """ One step of the RSSM: the next h, and z from the posterior when the frame's encoding (embed) is given,
    otherwise from the prior (its most likely z). Returns h, z and the prior's and the posterior's outputs (None
    without embed). """
    h = self.gru(torch.cat((z, action), -1), h)
    prior = self.prior_net(h)
    posterior = self.posterior_net(torch.cat((h, embed), -1)) if embed is not None else None
    z = self.sample_stochastic(posterior, training) if embed is not None else self.sample_stochastic(prior, training=False)
    return h, z, prior, posterior

  def decode(self, h, z):
    return self.decoder_net(self.decoder_input_net(torch.cat((h, z), -1)).unflatten(1, self.enc_shape))

  def forward(self, X_text, X_images, targets=None, actions=None, pose=None, frames=None, sequence_actions=None, rewards=None):
    """ FM.forward (the policy's action loss on the chunk, actions), plus the world model's losses on the current frame
    and the n_steps frames after it (frames, after each of sequence_actions, with rewards): the frames decoded from the
    posterior's latents, DreamerV3's KL losses, the reward and the critic's losses. Returns the class token's features when targets is False (from act()),
    otherwise (logits, loss, parts). """
    if targets is False:
        return self.fm(X_text, X_images, targets=False, pose=pose)
    logits, fm_loss, parts = self.fm(X_text, X_images, targets, actions, pose)
    world_model, world_parts, start = self.world_model_loss(X_images, sequence_actions, frames, rewards)
    actor, actor_parts = self.actor_loss(X_text, *start) if self._plain_cfg.model.actor_coef > 0 else (0, {})
    m = self._plain_cfg.model
    return (logits, fm_loss + m.frame_loss_weight * world_model + m.actor_coef * actor, {**parts, **world_parts, **actor_parts})

  ## Run eagerly, outside the training's compiled graph: in bf16, torch 2.5.1's compiled backward of mini-FM and the world
  ## model in one graph reads memory it never initialized, which turns mini-FM's gradients to NaN in some runs (6 of 6
  ## replays of a captured batch, 0 of 6 with the world model eager).
  @torch.compiler.disable
  def world_model_loss(self, X_images, actions, frames, rewards):
    """ The world model's loss on the current frame and the next n_steps, and its parts: the frames decoded from the
    posterior's latents, DreamerV3's KL losses, the reward after each action (rewards: (B, n_steps)) and the critic's
    loss; and the state (h, z) of the current frame, where imagination starts. """
    cfg = self._plain_cfg
    ## The sequence: the current frame, with no action before it (zeros, as DreamerV3's first step), then the frame
    ## after each of the next n_steps actions: it learns p(s'|s, a) for the actions taken. Past an episode's end the
    ## last frame and action repeat
    observations = torch.cat((self.small_frames(X_images), self.small_frames(frames)), dim=1) ## (B, 1 + n_steps, 3, size, size)
    actions = actions.unflatten(1, (self.n_steps, -1))
    actions = torch.cat((torch.zeros_like(actions[:, :1]), actions), dim=1)
    embed = self.encoder(observations.flatten(0, 1)).unflatten(0, observations.shape[:2])
    B = len(observations)
    h, z = observations.new_zeros(B, cfg.model.deter_dim), observations.new_zeros(B, self.z_dim)
    reconstructions, dyn_losses, rep_losses, features = [], [], [], []
    for t in range(observations.shape[1]):
        h, z, prior, posterior = self.rssm_step(h, z, actions[:, t], embed[:, t])
        if t == 0:
            start = (h.detach(), z.detach())
        features.append(torch.cat((h, z), -1))
        reconstructions.append(self.decode(h, z))
        ## DreamerV3's KL balancing, each with free bits (no pull below kl_free_nats): the dynamics loss trains the
        ## prior towards the posterior, the representation loss (weighted less) the posterior towards the prior
        dyn_losses.append(self.kl(posterior.detach(), prior).clamp(min=cfg.model.kl_free_nats))
        rep_losses.append(self.kl(posterior, prior.detach()).clamp(min=cfg.model.kl_free_nats))
    ## The frames' squared error summed over each frame's pixels, as DreamerV3's log-likelihood, so it weighs against
    ## the KLs (summed over z's variables) as in the paper; with a mean over pixels the KLs win and z learns nothing.
    ## frames_mse, the mean per pixel, is logged too
    squared_error = (torch.stack(reconstructions, 1).float() - observations) ** 2
    parts = {"frames": squared_error.sum((-3, -2, -1)).mean(), "frames_mse": squared_error.mean(),
             "kl_dyn": torch.stack(dyn_losses).mean(), "kl_rep": torch.stack(rep_losses).mean()}
    ## The reward of each action, predicted from the state it leads to
    features = torch.stack(features, 1) ## (B, 1 + n_steps, deter_dim + z_dim)
    parts["reward"] = ((self.reward_head(features[:, 1:]).squeeze(-1).float() - symlog(rewards)) ** 2).mean()
    critic, critic_parts = self.critic_loss(features, rewards)
    parts.update(critic_parts)
    ## parts stay unweighted for logging
    return (parts["frames"] + cfg.model.kl_dyn * parts["kl_dyn"] + cfg.model.kl_rep * parts["kl_rep"] + parts["reward"]
            + critic, parts, start)

  ## The actor loss with REINFORCE, for one imagined trajectory (H = model.imagine_horizon = 16). Only the bracketed
  ## [log π] terms carry gradients, into mini-FM; everything else is computed without them.
  ##
  ##   real o_0 ─ encoder, posterior ─> s_0 = (h_0, z_0)
  ##
  ##   s_0 ──decode──> ô_0 ──mini-FM──> μ_0 ── a_0 = μ_0 + std·ε ──prior──> s_1 ──decode──> ô_1 ──mini-FM──> … ──> s_16
  ##    │                                │                            │                                          │
  ##   v_0 (critic)            [log π(a_0|ô_0)]               r_1 (reward), v_1 (critic)          r_16, v_16 = V_16
  ##
  ##   returns, from the end back (DreamerV2's λ-target):  V_16 = v_16,  V_t = r_t+1 + γ((1 - λ) v_t+1 + λ V_t+1)
  ##                                                       (λ = model.return_lambda = 0.95, γ = model.return_discount = 0.99)
  ##   advantages:                                         A_t = V_t - v_t           (the critic's value is the baseline)
  ##   actor loss:                                         -mean over t and the batch of [log π(a_t|ô_t)] · A_t
  ##
  ## Notes:
  ## - log π(a|ô) is the log-density of a Gaussian of fixed std (model.actor_std) around mini-FM's prediction μ, up to a
  ##   constant: -|a - μ|² / (2 std²), summed over the action's 7 dimensions. Its gradient pulls μ towards actions with
  ##   a positive advantage and away from those with a negative one, by (a - μ) / std² = ε / std.
  ## - REINFORCE (DreamerV2's rho = 1) needs no gradients of the world model, only its rewards and values; DreamerV2
  ##   backpropagates through the dynamics instead (rho = 0) for continuous control, which would also steer mini-FM by
  ##   the world model's errors. Its entropy bonus is left out: with a fixed std the entropy is constant.
  ## - λ = 1 would be the 16-step return, the discounted rewards plus γ^(16-t) v_16 (less biased by the critic, noisier);
  ##   λ = 0 one reward plus the critic's next value (one-step TD: low variance, as biased as the critic). 0.95 mixes them.
  ## - One trajectory per batch row, from its current frame (512 per training step); the critic's baseline is all that
  ##   lowers the variance. The imitation loss, next to this one (which model.actor_coef weighs), keeps mini-FM near the
  ##   demonstrations where the world model is wrong.
  @torch.compiler.disable
  def actor_loss(self, X_text, h, z):
    """ mini-FM's loss in imagination: DreamerV2's actor loss with REINFORCE gradients (rho = 1; arXiv 2010.02193,
    "Actor loss"). From the states (h, z) of the real current frames, mini-FM acts on the decoded frame (upsampled to
    the camera images it trains on), sampling its action from a Gaussian around its prediction (model.actor_std), and
    the prior steps the state on that action, model.imagine_horizon (H) times. The world model runs without gradients:
    it only generates and scores the trajectories. Each step's return is DreamerV2's lambda-target, a mixture of the
    predicted rewards and the critic's values up to H, then the critic's value of the last state from there on:
    V_t = r_t+1 + discount * ((1 - lambda) * v(s_t+1) + lambda * V_t+1), V_H = v(s_H). Its advantage, the return minus
    the critic's value of the step's state (the baseline), weighs the log-probability of the action taken. """
    m = self._plain_cfg.model
    head, std, image_size = self.fm.action_head, m.actor_std, tuple(self._plain_cfg.image_shape[:2])
    log_probs, rewards, values = [], [], []
    for t in range(m.imagine_horizon + 1):
        with torch.no_grad():
            values.append(symexp(self.critic(torch.cat((h, z), -1)).squeeze(-1).float()))
            if t == m.imagine_horizon:
                break
            images = F.interpolate(self.decode(h, z), size=image_size, mode="bilinear").permute(0, 2, 3, 1)
        cond = self.fm(X_text, images, targets=False, pose=None)
        with torch.no_grad():
            action = head.sample(cond).float()
            action = action + std * torch.randn_like(action)
            h, z, _, _ = self.rssm_step(h, z, action.to(z.dtype), training=False)
            rewards.append(symexp(self.reward_head(torch.cat((h, z), -1)).squeeze(-1).float()))
        log_probs.append(head.log_prob(cond, action, std).sum(-1)) ## the Gaussian's, up to a constant
    with torch.no_grad(): ## rewards[t] follows action t, into the state of values[t + 1]
        returns = [values[-1]]
        for t in reversed(range(m.imagine_horizon)):
            returns.insert(0, rewards[t] + m.return_discount * ((1 - m.return_lambda) * values[t + 1] + m.return_lambda * returns[0]))
        returns = torch.stack(returns[:-1], 1) ## (B, horizon)
        advantages = returns - torch.stack(values[:-1], 1)
    loss = -(torch.stack(log_probs, 1) * advantages).mean()
    return loss, {"actor": loss, "imagined_return": returns[:, 0].mean(), "advantage": advantages.abs().mean()}

  def critic_loss(self, features, rewards):
    """ The critic's loss on the states of the sequence, features (B, 1 + n_steps, d): DreamerV3's lambda-return, of the
    real rewards and the slow critic's values, R_t = r_t+1 + discount * ((1 - lambda) * v(s_t+1) + lambda * R_t+1),
    with R_n = v(s_n), regressed in symlog space. The values are those of the demonstrations' actions. """
    m = self._plain_cfg.model
    with torch.no_grad():
        values = symexp(self.target_critic(features).squeeze(-1).float()) ## (B, 1 + n_steps)
        returns = [values[:, -1]]
        for t in reversed(range(self.n_steps)):
            returns.insert(0, rewards[:, t] + m.return_discount * ((1 - m.return_lambda) * values[:, t + 1] + m.return_lambda * returns[0]))
        returns = torch.stack(returns[:-1], 1) ## (B, n_steps)
    predicted = self.critic(features[:, :-1]).squeeze(-1).float()
    loss = ((predicted - symlog(returns)) ** 2).mean()
    return loss, {"critic": loss, "value": symexp(predicted).mean(), "return": returns.mean()}

  def act(self, X_text, X_images, pose):
    """ The (whitened) action chunk predicted for each row, flattened. """
    return self.fm.act(X_text, X_images, pose)

  @torch.no_grad()
  def frame_grid(self, batch, rows=4):
    """ An image of the predicted future, for logging: for each of the first rows samples of a batch, its current frame,
    its real next frames, then the ones the prior predicts for the actions taken from the current frame's state, all at
    frame_size. (H, W, 3) uint8. """
    X_text, X_images, _, _, pose, frames, actions, _ = (x[:rows] for x in batch)
    current = self.small_frames(X_images)
    actions = actions.unflatten(1, (self.n_steps, -1))
    h, z = current.new_zeros(rows, self._plain_cfg.model.deter_dim), current.new_zeros(rows, self.z_dim)
    h, z, _, _ = self.rssm_step(h, z, torch.zeros_like(actions[:, 0]), self.encoder(current[:, 0]), training=False)
    predicted = []
    for t in range(self.n_steps):
        h, z, _, _ = self.rssm_step(h, z, actions[:, t])
        predicted.append(self.decode(h, z))
    tiles = torch.cat((current, self.small_frames(frames), torch.stack(predicted, 1).float()), dim=1)
    low, high = self._cfg.image_bounds
    grid = rearrange(tiles.float(), 'r f c h w -> (r h) (f w) c')
    return ((grid - low) / (high - low) * 255).clamp(0, 255).byte().cpu().numpy()


def make_model(cfg):
  return Dreamer(cfg)


def make_policy(cfg): ## for BackgroundSimEval, which builds its own copy of the policy
    return FMPolicy(make_model(cfg), cfg)

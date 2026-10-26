## mini-VLAPS: VLAPS (Neary et al. 2025, arXiv 2508.12211), a search at test time that makes a pre-trained policy
## better, with no training: Monte Carlo tree search over action chunks in a world model, guided by the policy. Both
## come from one mini-Dreamer checkpoint (mini_dreamer.py, saved by mini_train.py): its RSSM is the world model, and the
## mini-FM trained next to it is the policy. Before each chunk runs in the sim, the search repeats vlaps.simulations times:
## - select: from the root, go to the child with the highest prior * sqrt(N(node) + 1) / (1 + N(child)), until a node
##   that hasn't been expanded: PUCT without its value term, as VLAPS, so the policy's prior and the visit counts alone
##   steer the search;
## - expand: the node's vlaps.children candidate chunks are the policy's chunk at the node's (imagined) frame, and the
##   same plus Gaussian noise (VLAPS samples them from a library of ~2000 chunks of successful runs, by their distance to
##   the policy's). Each gets a prior, a softmax of its negative distance to the policy's chunk mixed with uniform, and
##   the world model steps through it to a child;
## - roll out: each new child runs the policy in the world model for vlaps.rollout_chunks chunks, the policy reading
##   the frames decoded from the RSSM.
## The reward is sparse, as VLAPS's: success, once an imagined frame is within vlaps.goal_mse of the goal image (VLAPS
## has the simulator's success signal; a world model has none). The goal image of an episode is the last frame of the
## demonstration that started from the same init state (dataset_init_states). As soon as the search imagines a success,
## the first chunk of the path to it runs; otherwise the root's most visited chunk does. The search then starts again
## from the next real frame (VLAPS runs the whole path to the success, open loop). vlaps.simulations=0 runs mini-FM
## alone, the baseline the search should beat.
##   uv run mini_train.py --config-name mini-vlaps init_model=<path to miniDreamer_best.pth>
import logging
import os
import time
import numpy as np
import torch
from torch.nn import functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
import mini_dreamer
from mini_fm import FMPolicy
from mini_fm_rl import libero_settings
from sim_eval import make_libero_pool, rollout_libero, image_key_of, dataset_init_states, dataset_goal_images, LIBERO_CAMERAS
from logger import Logger
log = logging.getLogger(__name__) ## hydra writes it to the run folder's mini_train.log


class Node:
  """ A state of the search tree: the world model's state (h, z) and its frame. Once expanded: its candidate chunks,
  their priors and visit counts, and the children they lead to. """
  def __init__(self, h, z, frame):
    self.h, self.z, self.frame = h, z, frame
    self.chunks = self.prior = self.visits = self.children = None


class VLAPS:
  """ mini-FM, searching in mini-Dreamer's world model (dreamer.fm and dreamer, one checkpoint). act() has mini-FM's
  interface, so FMPolicy runs it in the sims. goal: the goal image of the episode, (1, 3, frame_size, frame_size). """
  def __init__(self, dreamer, cfg):
    self.dreamer, self.fm, self.cfg, self._cfg = dreamer, dreamer.fm, cfg, dreamer._cfg
    self.goal, self.successes, self.searches = None, 0, 0 ## the searches that imagined a success, of all searches

  def imagine(self, h, z, chunks):
    """ Steps the world model through action chunks (n, chunk * action_dim), one action at a time: the states after
    them and their decoded frames. """
    for action in chunks.unflatten(1, (self._cfg.policy.action_stacking, -1)).unbind(1):
        h, z, _, _ = self.dreamer.rssm_step(h, z, action)
    return h, z, self.dreamer.decode(h, z).float()

  def policy(self, X_text, frames):
    """ mini-FM's chunks at imagined frames (n, 3, frame_size, frame_size), upsampled to the images it was trained on. """
    images = F.interpolate(frames, size=tuple(self._cfg.image_shape[:2]), mode="bilinear").permute(0, 2, 3, 1)
    return self.fm.act(X_text.expand(len(frames), -1), images, None).float()

  def reached_goal(self, frames):
    return ((frames - self.goal) ** 2).mean((1, 2, 3)) < self.cfg.goal_mse

  def expand(self, node, X_text):
    """ Gives the node its candidate chunks, their priors and the children they lead to, then rolls every child out
    with the policy. Returns the child from which the goal was reached (imagined), or None. """
    c = self.cfg
    policy_chunk = self.policy(X_text, node.frame)
    node.chunks = torch.cat((policy_chunk, policy_chunk + c.noise_std * torch.randn(c.children - 1, policy_chunk.shape[1], device=policy_chunk.device)))
    distance = (node.chunks - policy_chunk).norm(dim=1)
    node.prior = (1 - c.epsilon) * (-c.alpha * distance).softmax(0) + c.epsilon / c.children
    node.visits = torch.zeros(c.children, device=policy_chunk.device)
    h, z, frames = self.imagine(node.h.expand(c.children, -1), node.z.expand(c.children, -1), node.chunks)
    node.children = [Node(h[i:i + 1], z[i:i + 1], frames[i:i + 1]) for i in range(c.children)]
    reached = self.reached_goal(frames)
    for _ in range(c.rollout_chunks): ## all the children's rollouts at once
        h, z, frames = self.imagine(h, z, self.policy(X_text, frames))
        reached |= self.reached_goal(frames)
    return int(reached.float().argmax()) if reached.any() else None

  @torch.no_grad()
  def search(self, X_text, X_image):
    """ The chunk to run from one real frame X_image (1, H, W, 3), with goal text X_text (1, max_block_size). """
    d = self.dreamer
    current = d.small_frames(X_image)[:, 0]
    h, z = current.new_zeros(1, self._cfg.model.deter_dim), current.new_zeros(1, d.z_dim)
    h, z, _, _ = d.rssm_step(h, z, current.new_zeros(1, self._cfg.action_dim), d.encoder(current), training=False) ## the posterior, as mini_dreamer's frame_grid
    root = Node(h, z, current)
    self.searches += 1
    for _ in range(self.cfg.simulations):
        node, path = root, [] ## path: the (node, child) pairs selected from the root
        while node.children is not None:
            child = int((node.prior * (node.visits.sum() + 1).sqrt() / (1 + node.visits)).argmax())
            path.append((node, child))
            node = node.children[child]
        reached = self.expand(node, X_text)
        for parent, child in path:
            parent.visits[child] += 1
        if reached is not None:
            self.successes += 1
            return root.chunks[path[0][1] if path else reached][None]
    return root.chunks[root.visits.argmax()][None]

  def act(self, X_text, X_images, pose):
    """ The (whitened) action chunk for each row, flattened: searched for, or with vlaps.simulations=0 mini-FM's. """
    if self.cfg.simulations == 0:
        return self.fm.act(X_text, X_images, pose)
    return torch.cat([self.search(X_text[i:i + 1], X_images[i:i + 1].float()) for i in range(len(X_images))])


def load_dreamer(path):
  """ A mini-Dreamer saved by mini_train.py (torch.save of the whole model), rebuilt with the current code. Its saved
  config is merged over the defaults in conf/mini-fm.yaml, for keys added since it was saved. """
  from omegaconf import OmegaConf
  saved = torch.load(path, map_location="cpu", weights_only=False)
  cfg = OmegaConf.merge(OmegaConf.load(os.path.join(HERE, "conf", "mini-fm.yaml")), saved._cfg)
  OmegaConf.set_struct(cfg, False) ## run() adds sim.obs_key
  assert cfg.model.type == "dreamer", f"{path} is a {cfg.model.type}, not a mini-Dreamer"
  model = mini_dreamer.make_model(cfg)
  model.load_state_dict(saved.state_dict())
  return model, cfg


def run(cfg):
  """ VLAPS in a LIBERO suite (conf/mini-vlaps.yaml), started by mini_train.py: one episode for each of
  cfg.init_states of each task, searching towards the last frame of the demo that started from that init state. """
  from libero.libero import benchmark
  torch.manual_seed(cfg.r_seed)
  dreamer, model_cfg = load_dreamer(os.path.join(HERE, cfg.init_model)) ## relative to the repo root
  model_cfg.device = cfg.device
  dreamer.to(cfg.device).eval()
  assert not model_cfg.policy.use_pose_data, "the world model doesn't predict the pose, so the policy can't use it"
  vlaps = VLAPS(dreamer, cfg.vlaps)
  policy = FMPolicy(vlaps, model_cfg)
  suite_name, episode_length, wait_steps = libero_settings(cfg)
  policy.task_set = suite_name ## un-whiten its actions with the stats of that suite's dataset
  key = image_key_of(model_cfg, suite_name) ## and render the camera it was trained on
  model_cfg.sim.obs_key = LIBERO_CAMERAS[key]
  amp_dtype = {"bf16": torch.bfloat16}.get(cfg.precision)
  logger = Logger(cfg)
  suite = benchmark.get_benchmark_dict()[suite_name]()
  tasks = list(cfg.tasks) if cfg.tasks is not None else list(range(suite.n_tasks))
  init_states = dataset_init_states(model_cfg, suite, suite_name)
  envs, env_tasks = make_libero_pool(suite, tasks, 1, model_cfg.sim.camera_size, 1)
  os.makedirs("eval", exist_ok=True)
  results = []
  for t in tasks:
    for i in range(*cfg.init_states):
        ## The goal image: the last frame of the demo that started from init state i, at frame_size
        goal = torch.tensor(dataset_goal_images(model_cfg, suite, suite_name, [(t, i)])[0], device=cfg.device)
        vlaps.goal = dreamer.small_frames(goal[None])[:, 0]
        vlaps.successes = vlaps.searches = 0
        t0 = time.time()
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            success, _, videos, _, _ = rollout_libero(policy, suite, envs, env_tasks, [(t, i)], episode_length, wait_steps, init_states=init_states)
        results.append(bool(success[0]))
        imagined = vlaps.successes / max(vlaps.searches, 1)
        log.info(f"task {t} init state {i}: success {results[-1]}, {vlaps.searches} searches, {imagined:.2f} of them "
                 f"imagined reaching the goal, {time.time() - t0:.0f}s")
        import imageio
        imageio.mimsave(f"eval/task_{t}_init_{i}.mp4", videos[0], fps=20)
        logger.log({"vlaps/success": float(results[-1]), "vlaps/imagined_success": imagined, "vlaps/episode_seconds": time.time() - t0}, len(results))
  log.info(f"VLAPS success rate {np.mean(results):.2f} over {len(results)} episodes of {suite_name}")
  logger.log({"vlaps/success_rate": float(np.mean(results))}, len(results))
  envs.close()
  logger.close()

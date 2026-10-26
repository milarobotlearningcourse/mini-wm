## RL fine-tuning of a pretrained mini-FM policy (the miniFM_best.pth saved by mini_train.py) on success in LIBERO, with
## GRPO (Shao et al. 2024) or PPO (Schulman et al. 2017) (cfg.algorithm). The policy is a Gaussian around the MSE head's
## predicted action chunk with a fixed std (cfg.action_std), so its log-probability is the MSE loss scaled by
## -1 / (2 std^2). Each iteration runs groups_per_task groups of rollouts on each of tasks_per_iter different tasks, each
## group group_size episodes from one init state. The advantage of each query (action chunk) is
## - GRPO: its episode's success against the mean and std of its group, with no learned value function;
## - PPO: GAE over its episode, from a reward of 1 when the task succeeds and a learned value function (value head).
## Both update the policy with PPO's clipped objective, plus an optional KL penalty that keeps each update near the policy
## before it (the one that collected the rollouts).
##   python mini_train.py --config-name mini-fm-rl init_model=<path to miniFM_best.pth> algorithm=ppo
import logging
import os
import time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
from mini_fm import make_model, FMPolicy
from data_loading import n_cameras
from sim_eval import make_libero_pool, rollout_libero, image_key_of, eval_task_sets, LIBERO_CAMERAS, LIBERO_MAX_STEPS ## the rollouts, with the same policy interface as the sim evals
from logger import Logger
log = logging.getLogger(__name__) ## hydra writes it to the run folder's mini_train.log


def load_pretrained(path):
    """ A model saved by mini_train.py (torch.save of the whole model), rebuilt with the current code. Its saved config
    is merged over the defaults in conf/mini-fm.yaml, for keys added since it was saved; then its weights are loaded. """
    from omegaconf import OmegaConf
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.merge(OmegaConf.load(os.path.join(HERE, "conf", "mini-fm.yaml")), saved._cfg)
    OmegaConf.set_struct(cfg, False) ## train() adds sim.obs_key, which models saved with hydra's config groups (sim/) refuse
    assert cfg.dataset.action_mean is not None, f"{path} is from before mini_fm.py stored the action stats in its config"
    assert cfg.model.action_head in ("mse", "l1", "huber"), "RL fine-tuning needs a regression head: its prediction is the Gaussian policy's mean"
    model = make_model(cfg)
    missing, unexpected = model.load_state_dict(saved.state_dict(), strict=False)
    assert not missing, f"weights missing from {path}: {missing}"
    print(f"loaded {path}: model.type {cfg.model.type}" + (f", ignored weights of removed layers {unexpected}" if unexpected else ""))
    return model, cfg


def timing_stats(prefix, timing, seconds, n_queries, n_envs, it):
    """ rollout_libero's timing, logged to mini_train.log and as perf/<prefix>_*: the rollout's seconds, the main
    process's seconds in policy queries (of which model_seconds in the model's inference) and waiting for envs, the envs'
    seconds simulating, the envs per policy query, and sim_utilization: the share of the envs' time spent simulating (1
    means they never wait for the policy). """
    sim_utilization = timing["sim_seconds"] / (n_envs * seconds)
    log.info(f"iter {it}: {prefix} {seconds:.1f}s: model inference {timing['model_seconds']:.1f}s, the rest of the policy queries "
             f"{timing['policy_seconds'] - timing['model_seconds']:.1f}s, waiting for the envs {timing['env_wait_seconds']:.1f}s; "
             f"{n_envs} envs simulating {sim_utilization:.0%} of the time, {n_queries} queries")
    return {f"perf/{prefix}_seconds": seconds, **{f"perf/{prefix}_{k}": v for k, v in timing.items()},
            f"perf/{prefix}_policy_batch": n_queries / max(timing["policy_calls"], 1),
            f"perf/{prefix}_sim_utilization": sim_utilization}


def libero_settings(cfg):
    """ The LIBERO suite to train in, its max policy steps per episode and its settling no-ops: from the sim config
    (conf/sim/, e.g. sim=libero90), the same settings as the pre-training sim evals. """
    task_sets = eval_task_sets(cfg)
    assert len(task_sets) == 1 and task_sets[0] in LIBERO_MAX_STEPS, f"RL trains in one LIBERO suite: pick it with sim=<config> or sim.task_sets=[<suite>], got {task_sets}"
    libero = cfg.sim.simulators.libero
    return task_sets[0], libero.episode_length or LIBERO_MAX_STEPS[task_sets[0]], libero.wait_steps


def evaluate(policy, suite, envs, env_tasks, tasks, cfg, it, logger):
    """ The success rate of the policy's predicted chunks (no noise) on cfg.eval_init_states of every task, which
    training never samples. Saves a video of the first episode. """
    import imageio
    jobs = [(t, i) for t in tasks for i in range(*cfg.eval_init_states)]
    t0 = time.time()
    _, episode_length, wait_steps = libero_settings(cfg)
    success, queries, videos, timing, _ = rollout_libero(policy, suite, envs, env_tasks, jobs, episode_length, wait_steps)
    seconds = time.time() - t0
    os.makedirs("eval", exist_ok=True)
    imageio.mimsave(f"eval/iter_{it}_task_{jobs[0][0]}_init_{jobs[0][1]}.mp4", videos[0], fps=20)
    per_task = {t: success[[j for j, (tj, _) in enumerate(jobs) if tj == t]].mean() for t in tasks}
    log.info(f"iter {it}: eval success {success.mean():.2f} over {len(jobs)} episodes, per task "
             + " ".join(f"{t}:{r:.2f}" for t, r in per_task.items()))
    logger.log({"eval/success_rate": float(success.mean()), **timing_stats("eval", timing, seconds, len(queries["episode"]), len(env_tasks), it)}, it,
               tensorboard_only={f"eval_task/{t}": float(r) for t, r in per_task.items()})


def features(model, queries, idx):
    """ The policy's features for queries idx, which its action head and PPO's value head read. """
    return model(queries["txt"][idx], queries["obs"][idx], targets=False, pose=queries["pose"][idx] if queries["pose"] is not None else None)


def make_value_head(model, model_cfg):
    """ PPO's critic: an MLP from the policy's features (FM's class token, or SmallConvNetFM's hidden layer) to the
    value, the expected discounted success. It trains on detached features, so it never changes the policy. """
    device = model_cfg.device
    pose = torch.zeros(1, len(model_cfg.dataset.pose_mean), device=device) if model_cfg.policy.use_pose_data else None
    height, width, channels = model_cfg.image_shape
    image = torch.zeros(1, height, width * n_cameras(model_cfg), channels, device=device) ## the cameras side by side
    with torch.no_grad():
        size = model(torch.zeros(1, model_cfg.max_block_size, dtype=torch.long, device=device),
                     image, targets=False, pose=pose).shape[-1]
    return torch.nn.Sequential(torch.nn.Linear(size, 256), torch.nn.Tanh(), torch.nn.Linear(256, 1)).to(device)


def gae(model, value_head, queries, success, cfg, autocast):
    """ PPO's advantage and return of each query: GAE (Schulman et al. 2016) over each episode's queries (its action
    chunks, in order), with a reward of 1 on the chunk where the task succeeds and 0 otherwise. Success and the time
    limit both end an episode with a value of 0 after it, so the value is the discounted chance of succeeding in time. """
    with torch.no_grad(), autocast():
        values = torch.cat([value_head(features(model, queries, idx).float()).float().squeeze(-1).cpu()
                            for idx in torch.arange(len(queries["action"]), device=cfg.device).split(cfg.minibatch_size)])
    advantages = torch.zeros_like(values)
    for e in queries["episode"].unique(): ## rollout_libero records each episode's queries in order
        idx = (queries["episode"] == e).nonzero().flatten()
        v = values[idx]
        reward = torch.zeros_like(v)
        reward[-1] = float(success[int(e)])
        delta = reward + cfg.gamma * torch.cat([v[1:], v.new_zeros(1)]) - v ## TD errors
        a = 0.0
        for t in reversed(range(len(v))):
            a = delta[t] + cfg.gamma * cfg.gae_lambda * a
            advantages[idx[t]] = a
    return advantages, advantages + values


def policy_update(model, optimizer, queries, advantages, cfg, autocast, value_head=None, value_optimizer=None, returns=None):
    """ The update on one iteration's rollouts, given the advantage of each query (GRPO's or PPO's). As in RLinf's RL
    for VLAs, PPO's clipped objective clips the ratio per action value (each dimension of each step of a chunk, like
    a VLA's action tokens) and the loss is their mean. Each query is weighted by 1 / its episode's number of queries, so
    every episode counts the same. Queries with advantage 0 give no signal (in GRPO, groups where every rollout had the
    same outcome) and are left out. With a value head (PPO), it is also fit to the returns. Returns stats for logging. """
    keep = advantages != 0
    if not keep.any():
        return {"used_queries": 0.0}
    q = {k: v[keep.to(v.device)] if v is not None else None for k, v in queries.items()}
    episode = q["episode"].to(cfg.device)
    A = advantages[keep].to(cfg.device)[:, None] ## the same for every action value of the query
    weight = 1.0 / torch.bincount(episode)[episode].float()
    returns = returns[keep].to(cfg.device) if returns is not None else None
    head, std = model.action_head, cfg.action_std
    batches = lambda: torch.randperm(len(A), device=cfg.device).split(cfg.minibatch_size)
    with torch.no_grad(), autocast(): ## the mean chunk of the policy that acted: the weights before this update
        old_mean = torch.cat([head.sample(features(model, q, idx)).float() for idx in torch.arange(len(A), device=cfg.device).split(cfg.minibatch_size)])
    old_logp = -(old_mean - q["action"]) ** 2 / (2 * std ** 2) ## as head.log_prob
    stats = {"loss": [], "kl": [], "clip_frac": [], "value_loss": []}
    for _ in range(cfg.epochs):
        for idx in batches():
            with autocast():
                cond = features(model, q, idx)
                logp = head.log_prob(cond, q["action"][idx], std)
                ## KL to the policy before this update: for two Gaussians with the same std it is
                ## |mean - old_mean|^2 / (2 std^2), which is minus the log-prob of old_mean under the current policy
                kl = -head.log_prob(cond, old_mean[idx], std)
            ratio = torch.exp(logp - old_logp[idx])
            surrogate = torch.min(ratio * A[idx], ratio.clamp(1 - cfg.clip_low, 1 + cfg.clip_high) * A[idx])
            w = weight[idx]
            loss = -(w * (surrogate - cfg.kl_weight * kl).mean(-1)).sum() / w.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            stats["loss"].append(loss.item()); stats["kl"].append(kl.sum(-1).mean().item()) ## the KL of whole chunks
            stats["clip_frac"].append(((ratio < 1 - cfg.clip_low) | (ratio > 1 + cfg.clip_high)).float().mean().item())
            if value_head is not None:
                value_loss = ((value_head(cond.detach().float()).squeeze(-1) - returns[idx]) ** 2).mean()
                value_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                value_optimizer.step()
                stats["value_loss"].append(value_loss.item())
    return {"used_queries": float(keep.float().mean()), **{k: float(np.mean(v)) for k, v in stats.items() if v}}


def train(cfg):
    """ The RL fine-tuning run of cfg (conf/mini-fm-rl.yaml), started by mini_train.py. """
    from libero.libero import benchmark
    torch.manual_seed(cfg.r_seed)
    rng = np.random.default_rng(cfg.r_seed)
    model, model_cfg = load_pretrained(os.path.join(HERE, cfg.init_model)) ## relative to the repo root
    model_cfg.device = cfg.device
    ## Eval mode (no dropout) in the updates too, so their log-probs are of the same policy that acted
    model.to(cfg.device).eval()
    policy = FMPolicy(model, model_cfg)
    suite_name, episode_length, wait_steps = libero_settings(cfg)
    policy.task_set = suite_name ## un-whiten its actions with the stats of that suite's dataset
    key = image_key_of(model_cfg, suite_name) ## and render the camera(s) it was trained on
    model_cfg.sim.obs_key = LIBERO_CAMERAS[key] if isinstance(key, str) else [LIBERO_CAMERAS[k] for k in key]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    assert cfg.algorithm in ("grpo", "ppo"), cfg.algorithm
    value_head = make_value_head(model, model_cfg) if cfg.algorithm == "ppo" else None
    value_optimizer = torch.optim.AdamW(value_head.parameters(), lr=cfg.value_learning_rate) if value_head is not None else None
    amp_dtype = {"bf16": torch.bfloat16}.get(cfg.precision)
    autocast = lambda: torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None)
    logger = Logger(cfg)
    suite = benchmark.get_benchmark_dict()[suite_name]()
    tasks = list(cfg.tasks) if cfg.tasks is not None else list(range(suite.n_tasks))
    ## Open for the whole run: building the envs costs more than their soft resets between episodes
    envs, env_tasks = make_libero_pool(suite, tasks, cfg.envs_per_task, model_cfg.sim.camera_size, cfg.min_ready_envs)
    for it in range(cfg.iterations + 1):
        if it % cfg.eval_interval == 0 or it == cfg.iterations:
            with autocast():
                evaluate(policy, suite, envs, env_tasks, tasks, cfg, it, logger)
            torch.save(model, "miniFM_rl.pth") ## at every eval, so a run stopped by its time limit keeps its latest model
        if it == cfg.iterations:
            break
        ## Collect: groups_per_task groups from each of tasks_per_iter different random tasks, each group group_size
        ## rollouts from one random training init state (different for each group of a task)
        t0 = time.time()
        groups = [(int(t), int(i)) for t in rng.choice(tasks, cfg.tasks_per_iter, replace=False)
                  for i in rng.choice(np.arange(*cfg.train_init_states), cfg.groups_per_task, replace=False)]
        with autocast():
            success, queries, _, timing, _ = rollout_libero(policy, suite, envs, env_tasks, [g for g in groups for _ in range(cfg.group_size)],
                                                 episode_length, wait_steps, cfg.action_std)
        t1 = time.time() ## the update, from here, includes the advantages (PPO's value head)
        reward = torch.tensor(success, dtype=torch.float32).view(len(groups), cfg.group_size)
        returns = None
        if cfg.algorithm == "grpo": ## each rollout's success against the mean and std of its group, for all its queries
            advantages = ((reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-6)).flatten()[queries["episode"]]
        else: ## GAE with the value head, normalized over the iteration's queries
            advantages, returns = gae(model, value_head, queries, success, cfg, autocast)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        stats = policy_update(model, optimizer, queries, advantages, cfg, autocast, value_head, value_optimizer, returns)
        t2 = time.time() ## policy_update's stats (.item()) already waited for the GPU
        group_success = reward.mean(1)
        log.info(f"iter {it}: {len(success)} rollouts, success {success.mean():.2f}, groups with mixed outcomes "
                 f"{int(((group_success > 0) & (group_success < 1)).sum())}/{len(groups)}, " + ", ".join(f"{k} {v:.4f}" for k, v in stats.items()) + f", update {t2 - t1:.1f}s")
        logger.log({"train/rollout_success": float(success.mean()), **{f"train/{k}": v for k, v in stats.items()},
                    **timing_stats("rollout", timing, t1 - t0, len(queries["episode"]), len(env_tasks), it),
                    "perf/update_seconds": t2 - t1}, it)
    envs.close()
    print("Model saved to miniFM_rl.pth")
    logger.close()


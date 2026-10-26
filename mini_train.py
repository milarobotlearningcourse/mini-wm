## Trains the small robot foundation models of this folder, picked by the config:
## - pre-training (trainer: pretrain) of mini-FM (mini_fm.py: model.type fm or small_convnet) mini-WAM (mini_wam.py:
##   model.type wam), mini-Dreamer (mini_dreamer.py: model.type dreamer, a forward model with mini-FM as its policy) or
##   mini-JEPA (mini_jepa.py: model.type jepa, a latent world model that plans towards a goal image, or with mini-FM as its policy) or mini-TD-MPC2
##   (mini_tdmpc2.py: model.type tdmpc2, a world model that plans its actions, learned offline), on robot data, with sim evals during training;
## - RL fine-tuning in LIBERO (trainer: rl) of a pre-trained mini-FM, with GRPO or PPO (mini_fm_rl.py);
## - no training (trainer: vlaps): Monte Carlo tree search with a pre-trained mini-Dreamer and its mini-FM, as VLAPS (mini_vlaps.py).
##   uv run mini_train.py                           (mini-FM, conf/mini-fm.yaml)
##   uv run mini_train.py --config-name mini-wam    (mini-WAM, conf/mini-wam.yaml)
##   uv run mini_train.py --config-name mini-dreamer  (mini-Dreamer, conf/mini-dreamer.yaml)
##   uv run mini_train.py --config-name mini-jepa   (mini-JEPA, conf/mini-jepa.yaml)
##   uv run mini_train.py --config-name mini-tdmpc2 (mini-TD-MPC2, conf/mini-tdmpc2.yaml)
##   uv run mini_train.py --config-name mini-fm-rl  (RL fine-tuning, conf/mini-fm-rl.yaml)
## The data loading is in data_loading.py and the sim evals in sim_eval.py.
import logging
import os
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
import mini_fm, mini_wam, mini_dreamer, mini_jepa, mini_tdmpc2
from sim_eval import eval_sim_suites, eval_task_sets, log_sim_eval, BackgroundSimEval
from logger import Logger
from data_loading import (build_vocab, load_lerobots, episode_probabilities, RobotEpisodes, RobotCollate, RobotCache,
                          make_cache_loader, make_loader, to_device, n_cameras, POSE_KEY)

## hydra writes these lines to the run folder's mini_train.log as well as to the console (print only reaches the console)
log = logging.getLogger(__name__)
MODELS = {"fm": mini_fm, "small_convnet": mini_fm, "wam": mini_wam, "dreamer": mini_dreamer, "jepa": mini_jepa, "tdmpc2": mini_tdmpc2} ## the file of each model.type, with its make_model and make_policy


@torch.no_grad()
def estimate_loss(model, loaders, cfg):
    ## Returns {split: {"loss": total, "ce": token head, "mse", "l1", "huber" or "diffusion": action head, "frames": WAM's frame
    ## diffusion or Dreamer's decoded frames (summed over pixels; "frames_mse" per pixel), "kl_dyn", "kl_rep", "reward", "critic", "value",
    ## "return", "actor" and "imagined_return": Dreamer's,
    ## "latent_tf", "latent_ar", "latent_copy" and "target_std": JEPA's, "consistency", "reward_ce", "value_ce", "pi_loss", "q" and
    ## "reward_rate": TD-MPC2's}}; a head with no rows in the batch is left out.
    out = {}
    model.eval()
    for split, loader in loaders.items():
        sums = {}
        for k in range(cfg.eval_iters):
            batch = to_device(next(loader), cfg)
            logits, loss, parts = model(*batch)
            is_robot = batch[2] < 0
            has_rows = lambda name: (~is_robot).any().item() if name == "ce" else is_robot.any().item() ## every head but ce is on robot rows
            for name, value in {"loss": loss, **{n: v for n, v in parts.items() if has_rows(n)}}.items():
                sums[name] = sums.get(name, 0.0) + value.item()
        out[split] = {name: total / cfg.eval_iters for name, total in sums.items()}
    model.train()
    return out


@torch.no_grad()
def estimate_dataset_loss(model, cache, cfg):
    ## The loss of each robot head, the action head's (mse or diffusion) and WAM's frames, on each robot dataset of a
    ## split's RobotCache, {head: {short dataset name: loss}}, from robot-only batches, to see which datasets the model
    ## doesn't generalize to.
    out = {}
    model.eval()
    k = cfg.dataset.robot_frames_per_episode
    for dataset in sorted(cache.dataset_p):
        sums = {}
        for _ in range(max(1, cfg.eval_iters // 20)):
            parts = model(*to_device(cache.sample(cfg.batch_size // k, k, dataset), cfg))[2]
            for name in (cfg.model.action_head, "frames"):
                if name in parts: ## .item() right away: with CUDA graphs, the next call overwrites a call's outputs
                    sums.setdefault(name, []).append(parts[name].item())
        for name, losses in sums.items():
            out.setdefault(name, {})[dataset.split("/")[-1]] = float(np.mean(losses))
    model.train()
    return out


def random_shift(images, pad):
    """ Training augmentation (as DrQ): shifts each image by up to pad pixels in x and y, repeating the edge pixels, so the
    model can't memorize episodes from exact pixel positions. images: (B, H, W, C) on the GPU, in one indexing op.
    Blank images stay blank. """
    if pad == 0:
        return images
    B, H, W, _ = images.shape
    padded = F.pad(images.permute(0, 3, 1, 2), (pad, pad, pad, pad), mode="replicate").permute(0, 2, 3, 1)
    dy, dx = torch.randint(0, 2 * pad + 1, (2, B, 1), device=images.device)
    rows = (torch.arange(H, device=images.device) + dy)[:, :, None] ## (B, H, 1)
    cols = (torch.arange(W, device=images.device) + dx)[:, None, :] ## (B, 1, W)
    return padded[torch.arange(B, device=images.device)[:, None, None], rows, cols]

def color_jitter(images, strength, bounds):
    """ Training augmentation for the colour gap between datasets and their sims (e.g. real photos and a sim's
    renders): random brightness, contrast and saturation (each x0.7-1.3 at strength 1) and a gain per colour
    channel (x0.9-1.1, a white-balance shift) for each image. images: (N, H, W, k * 3) with pixels in bounds; the k
    images stacked in the channels (WAM's future frames) get the same jitter. Every change is a scaling around 0 (the
    low bound), so a blank image stays exactly blank. """
    if strength == 0:
        return images
    low, high = bounds
    x = ((images - low) / (high - low)).unflatten(-1, (-1, 3)) ## (N, H, W, k, 3) in [0, 1]
    factor = lambda spread, channels=1: 1 + spread * strength * (2 * torch.rand(len(x), 1, 1, 1, channels, device=x.device) - 1)
    x = x * factor(0.3) ## brightness
    mean = x.mean(dim=(1, 2, 4), keepdim=True)
    x = (x - mean) * factor(0.3) + mean ## contrast
    gray = x.mean(dim=-1, keepdim=True)
    x = (x - gray) * factor(0.3) + gray ## saturation
    x = x * factor(0.1, channels=3) ## colour cast
    return x.clamp(0, 1).flatten(-2) * (high - low) + low

def augment(batch, cfg):
    """ random_shift and color_jitter of the batch's images; a WAM batch's future frames get the same as the image
    they follow. Each of several cameras side by side (dataset.cameras) is augmented on its own, so no pixels cross into
    the next one, and a blank (missing) camera stays blank. """
    X_text, X_images, targets, actions, pose, *future = batch
    frames, future_actions = future[:1], future[1:] ## the future frames (WAM, Dreamer, JEPA, TD-MPC2), their actions (Dreamer, JEPA, TD-MPC2) and rewards (Dreamer, TD-MPC2)
    cameras = n_cameras(cfg)
    images = torch.cat(torch.cat((X_images, *frames), dim=-1).chunk(cameras, dim=2), dim=0) ## the frames are stacked in the channels
    images = color_jitter(images, cfg.augment_color, cfg.image_bounds)
    shifted = torch.cat(random_shift(images, cfg.augment_shift).chunk(cameras, dim=0), dim=2)
    return (X_text, shifted[..., :X_images.shape[-1]], targets, actions, pose, *(shifted[..., X_images.shape[-1]:] for _ in frames), *future_actions)


import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

@hydra.main(config_path="./conf", config_name="mini-fm")
def my_main(cfg: DictConfig):
    if cfg.trainer == "rl":
        import mini_fm_rl
        return mini_fm_rl.train(cfg)
    if cfg.trainer == "vlaps":
        import mini_vlaps
        return mini_vlaps.run(cfg)
    assert cfg.trainer == "pretrain", cfg.trainer
    start_time = time.time()
    torch.manual_seed(cfg.r_seed)
    np.random.seed(cfg.r_seed)
    ## Details on setting up the compute device for training
    print ("cfg:", OmegaConf.to_yaml(cfg))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if (device == 'cuda') else "")
    cfg.device = device

    ## fp32: full precision; tf32: fp32 weights with TF32 tensor-core matmuls; bf16/fp16: autocast mixed precision.
    assert cfg.precision in ("fp32", "tf32", "bf16", "fp16"), cfg.precision
    torch.backends.cuda.matmul.allow_tf32 = cfg.precision != "fp32"
    torch.backends.cudnn.allow_tf32 = cfg.precision != "fp32"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.precision)
    autocast = lambda: torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None)
    scaler = torch.amp.GradScaler(device, enabled=cfg.precision == "fp16") ## fp16 needs loss scaling to avoid gradient underflow
    print("precision:", cfg.precision)

    logger = Logger(cfg) ## wandb (unless testing) and TensorBoard

    names = [cfg.dataset.robot_dataset_name] if isinstance(cfg.dataset.robot_dataset_name, str) else list(cfg.dataset.robot_dataset_name)
    ## The camera of each robot dataset: one key for all, or one per dataset, or with dataset.cameras the same list of
    ## cameras for all, side by side
    keys = [list(cfg.dataset.cameras)] * len(names) if cfg.dataset.cameras else cfg.dataset.robot_image_key
    cfg.dataset.robot_image_key = [keys] * len(names) if isinstance(keys, str) else list(keys)
    assert len(cfg.dataset.robot_image_key) == len(names), \
        f"dataset.robot_image_key {list(cfg.dataset.robot_image_key)} must be one camera, or one per robot dataset {names}"
    ## Seconds each startup stage takes, logged (startup/) once training starts
    startup, t = {}, time.time()
    frames, episodes, tasks = load_lerobots(names, cfg.dataset.robot_image_key)
    startup["load_datasets"], t = time.time() - t, time.time()
    ## The robot data mix: the probability of sampling from each dataset (equal by default), whatever its size
    cfg.dataset.robot_dataset_weights = list(cfg.dataset.robot_dataset_weights or [1 / len(names)] * len(names))
    assert len(cfg.dataset.robot_dataset_weights) == len(names) and abs(sum(cfg.dataset.robot_dataset_weights) - 1) < 1e-6, \
        f"dataset.robot_dataset_weights {cfg.dataset.robot_dataset_weights} must have one probability per robot dataset, summing to 1"
    robot_weights = dict(zip(names, cfg.dataset.robot_dataset_weights))
    print("robot data mix: " + ", ".join(f"{n.split('/')[-1]} {w:.3f}" for n, w in robot_weights.items()))
    ## Episodes are stored task by task, so shuffle them before splitting to cover every task in both
    ## splits. The split is by episode, so no trajectory has frames in both.
    episodes = episodes.sample(frac=1, random_state=cfg.r_seed).reset_index(drop=True)
    n_val = max(1, int(0.05 * len(episodes)))
    k = cfg.dataset.robot_frames_per_episode
    assert cfg.batch_size % k == 0, f"batch_size ({cfg.batch_size}) must be a multiple of robot_frames_per_episode ({k})"
    robot_episodes = {"train": episodes[n_val:], "val": episodes[:n_val]}
    robot_splits = {split: RobotEpisodes(frames, eps, k, cfg.policy.action_stacking) for split, eps in robot_episodes.items()}
    cfg.action_dim = len(frames[0]["action"])
    ## Whitening stats from the frames of the train episodes (reads only the action and pose columns). They are stored
    ## in cfg, which is saved with the model, so predictions can be mapped back with action * std + mean. Actions get
    ## the stats of their own dataset; action_mean/std, over all the robot data, are for task sets without a dataset.
    rows_of = lambda eps: np.concatenate([np.arange(s, s + n) for s, n in zip(eps["dataset_from_index"], eps["length"])])
    column = lambda rows, key: np.stack(frames.select(rows).with_format("arrow")[key].to_numpy(zero_copy_only=False)).astype(np.float64)
    mean_std = lambda x: {"mean": x.mean(0).tolist(), "std": np.maximum(x.std(0), 1e-6).tolist()}
    train_rows = rows_of(episodes[n_val:])
    actions = {repo_id: column(rows_of(eps), "action") for repo_id, eps in episodes[n_val:].groupby("dataset", sort=False)}
    cfg.dataset.action_stats = {repo_id: mean_std(a) for repo_id, a in actions.items()}
    all_actions = mean_std(np.concatenate(list(actions.values())))
    cfg.dataset.action_mean, cfg.dataset.action_std = all_actions["mean"], all_actions["std"]
    for repo_id, s in cfg.dataset.action_stats.items():
        print(f"action mean {repo_id}: {np.round(s['mean'], 5)}, std: {np.round(s['std'], 5)}")
    poses = column(train_rows, POSE_KEY)
    cfg.pose_dim = poses.shape[1]
    cfg.dataset.pose_mean, cfg.dataset.pose_std = poses.mean(0).tolist(), np.maximum(poses.std(0), 1e-6).tolist()
    print(f"pose mean: {np.round(cfg.dataset.pose_mean, 5)}, std: {np.round(cfg.dataset.pose_std, 5)}")
    print(f"robot train episodes: {len(episodes) - n_val} ({len(train_rows)} frames), val episodes: {n_val}, "
          f"tasks: {len(tasks)}, action_dim: {cfg.action_dim}")
    chars, encode_txt, decode_txt = build_vocab(list(cfg.dataset.chars_list))
    cfg.dataset.chars_list = chars ## the vocabulary FMPolicy.encode_text_goal uses in the sim eval
    startup["action_pose_stats"], t = time.time() - t, time.time()

    num_workers = cfg.dataset.get("num_workers", 4)
    assert cfg.dataset.cache in ("cuda", "cpu", "none"), cfg.dataset.cache
    cache_device = device if cfg.dataset.cache == "cuda" else "cpu"
    loaders, robot_caches = {}, {}
    for split in ["train", "val"]:
        workers = num_workers if split == "train" else (max(1, num_workers // 2) if num_workers > 0 else 0)
        ## A batch picks batch_size / k episodes (with replacement, so a small val split still fills a batch), then k frames in each
        n_ep = cfg.batch_size // cfg.dataset.robot_frames_per_episode
        if cfg.dataset.cache != "none":
            cache = RobotCache(cfg, encode_txt, frames, robot_episodes[split], tasks, cache_device,
                               cfg.dataset.cache_episodes, robot_weights,
                               future_steps={"wam": cfg.policy.action_stacking, "dreamer": cfg.model.sequence_steps,
                                             "jepa": cfg.model.sequence_steps,
                                             "tdmpc2": cfg.model.horizon * cfg.policy.action_stacking}.get(cfg.model.type, 0),
                               future_actions=cfg.model.type in ("dreamer", "jepa", "tdmpc2"), future_rewards=cfg.model.type in ("dreamer", "tdmpc2"))
            robot_caches[split] = cache
            startup[f"{split}_cache"], startup[f"{split}_cache_decode"], t = time.time() - t, cache.decode_s, time.time()
            loaders[split] = make_cache_loader(cache.sample, n_ep, cfg.dataset.robot_frames_per_episode)
        else:
            p = episode_probabilities(robot_episodes[split], robot_weights)
            sampler = torch.utils.data.WeightedRandomSampler(p, num_samples=max(len(p), n_ep), replacement=True)
            loaders[split] = make_loader(robot_splits[split], cfg, RobotCollate(cfg, encode_txt, tasks), workers, n_ep, sampler)
    train_loader, val_loader = loaders["train"], loaders["val"]
    print(f"batch: {cfg.batch_size} robot samples, cache: {cfg.dataset.cache}")
    for split, cache in robot_caches.items():
        print(f"robot {split} cache: {len(cache.length)} of {len(robot_episodes[split])} episodes"
              + (", swapping in the rest from disk" if len(cache.length) < len(robot_episodes[split]) else ""))
    if cfg.dataset.cache == "cuda":
        print(f"GPU memory used by the cached datasets: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    ## Start every loader's (persistent) workers now. Forking them later, once torch.compile's threads are running,
    ## can segfault a worker at random.
    batch = to_device(next(train_loader), cfg) ## (text, images, targets, actions, pose), and future frames for WAM, Dreamer and JEPA (and their actions for Dreamer and JEPA)
    next(val_loader)
    startup["first_batch"], startup["total"] = time.time() - t, time.time() - start_time
    log.info("startup: " + ", ".join(f"{name} {value:.1f}s" for name, value in startup.items()))
    logger.log({f"startup/{name}_s": value for name, value in startup.items()}, 0)

    if cfg.sim.eval_interval > 0:
        with open_dict(cfg):
            cfg.sim.camera_size = cfg.image_shape[0] ## the image size trained on (each eval uses its dataset's camera, image_key_of)
            cfg.sim.task_sets = eval_task_sets(cfg)

    cfg.vocab_size = len(chars)
    cfg.action_bins = len(chars) ## The model predicts the next character
    print("vocab_size:", cfg.vocab_size)

    assert cfg.model.type not in ("wam", "dreamer", "jepa", "tdmpc2") or cfg.dataset.cache != "none", "WAM's, Dreamer's, JEPA's and TD-MPC2's future frames come from the robot cache: dataset.cache cuda or cpu"
    model = MODELS[cfg.model.type].make_model(cfg)
    model.to(device)
    ## The whole forward compiled as one graph for the fixed batch shape, with CUDA graphs (mode reduce-overhead), which
    ## replay the step's kernels without launching each one from Python: 1.32x faster training than compiling each block alone
    ## at batch 256. Used for training and the loss estimates; the eager model is used everywhere else.
    torch._inductor.config.triton.descriptive_names = False ## short kernel names: the fused ops' names can exceed the 255-character file-name limit
    fast_model = torch.compile(model, mode="reduce-overhead", dynamic=False)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    background_sim = BackgroundSimEval(MODELS[cfg.model.type].make_policy, cfg) if cfg.sim.eval_interval > 0 and cfg.sim.background else None
    model_name = {"wam": "miniWAM", "dreamer": "miniDreamer", "jepa": "miniJEPA", "tdmpc2": "miniTDMPC2"}.get(cfg.model.type, "miniFM") ## the saved models are <model_name>.pth and <model_name>_best.pth
    best = {"success": -1.0}
    def save_if_best(iter, success, weights=None):
        """ Saves the model as miniFM_best.pth (miniWAM_best.pth) (the one to fine-tune, e.g. with mini_fm_rl.py) when its overall sim
        success, the mean over the task sets, beats the best so far. weights: those of the evaluated step, when the
        eval ran in the background while training went on. """
        if success <= best["success"]:
            return
        best.update(success=success, step=iter)
        if weights is not None:
            current = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(weights)
        torch.save(model, f"./{model_name}_best.pth")
        if weights is not None:
            model.load_state_dict(current)
        print(f"step {iter}: best sim success so far ({success:.2f}), saved to ./{model_name}_best.pth")
        logger.log({"sim/best_success_rate": success, "sim/best_step": iter}, iter)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, fused=device == "cuda") ## fused: one kernel for all the weights
    import torch.optim.lr_scheduler as lr_scheduler
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.max_iters)


    gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9 if device == 'cuda' else 0
    ## Memory from the model size: weights + gradients + AdamW's two moment buffers, all fp32 (autocast only casts activations).
    model_mem = 4 * sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"GPU memory: {gpu_mem:.2f} GB, model memory (params + grads + AdamW): {model_mem:.3f} GB")

    ## Throughput since the last log: GPU utilization samples (NVML, % of time a kernel was running; device wide, so other
    ## processes on the GPU count too) and the time the loop spent waiting for the next batch.
    util, data_time, sample_time, t_log, iter_log = [], 0.0, 0.0, time.time(), 0
    swaps_log = robot_caches["train"].swaps if "train" in robot_caches else 0
    for iter in range(cfg.max_iters):

        if iter % cfg.eval_interval == 0 or iter == cfg.max_iters - 1:
            with autocast():
                losses = estimate_loss(fast_model, {"train": train_loader, "val": val_loader}, cfg)
                dataset_losses = estimate_dataset_loss(fast_model, robot_caches["val"], cfg) if "val" in robot_caches else {}
            ## Peak memory since the start of training, split into the model's share and the batch's share (activations + inputs).
            ## (gpu_mem - model) / batch_per_sample is roughly the largest batch size that fits.
            mem = {}
            if device == 'cuda':
                peak = torch.cuda.max_memory_allocated(device) / 1e9
                mem = {"allocated": torch.cuda.memory_allocated(device) / 1e9, "peak_allocated": peak,
                       "peak_reserved": torch.cuda.max_memory_reserved(device) / 1e9, "model": model_mem,
                       "batch": peak - model_mem, "batch_per_sample": (peak - model_mem) / cfg.batch_size}
            perf = {}
            if iter > iter_log:
                elapsed = time.time() - t_log
                ## per batch: the time to sample it (from the cache, or the DataLoader's queue) and then to copy it to the GPU
                perf = {"steps_per_s": (iter - iter_log) / elapsed, "data_wait_frac": data_time / elapsed,
                        "batch_sample_ms": 1000 * sample_time / (iter - iter_log),
                        "batch_to_device_ms": 1000 * (data_time - sample_time) / (iter - iter_log)}
                if "train" in robot_caches and len(robot_caches["train"].length) < len(robot_episodes["train"]):
                    perf["cache_swaps_per_s"] = (robot_caches["train"].swaps - swaps_log) / elapsed
                if util:
                    perf["gpu_util_pct"] = float(np.mean(util))
            log.info(f"step {iter}: " + ", ".join(f"{split} {name} {value:.4f}" for split in losses for name, value in losses[split].items())
                  + f", time {time.time() - start_time:.1f}s"
                  + (f", {perf['steps_per_s']:.1f} steps/s, data wait {100 * perf['data_wait_frac']:.0f}% (sample "
                     f"{perf['batch_sample_ms']:.1f} ms, to GPU {perf['batch_to_device_ms']:.1f} ms per batch)" if perf else "")
                  + (f", gpu util {perf['gpu_util_pct']:.0f}%" if "gpu_util_pct" in perf else "")
                  + (f", cache swaps {perf['cache_swaps_per_s']:.1f}/s" if "cache_swaps_per_s" in perf else "")
                  + (f", mem peak {mem['peak_allocated']:.2f}/{gpu_mem:.2f} GB (model {model_mem:.3f}, batch {mem['batch']:.3f})" if mem else ""))
            images = {}
            if cfg.model.type in ("wam", "dreamer", "jepa") and (iter % cfg.model.frame_image_interval == 0 or iter == cfg.max_iters - 1):
                ## what it predicts the next frames of a few val samples look like
                model.eval()
                with autocast():
                    images["predicted_frames"] = model.frame_grid(to_device(next(val_loader), cfg))
                model.train()
                os.makedirs("predicted_frames", exist_ok=True)
                import imageio
                imageio.imwrite(f"predicted_frames/{iter:06d}.png", images["predicted_frames"])
            logger.log({**{f"{name}/{split}": value for split in losses for name, value in losses[split].items()}, ## train and val of each head share one plot
                        **{f"val_by_dataset/{name}": value for name, value in dataset_losses.get(cfg.model.action_head, {}).items()},
                        **{f"val_frames_by_dataset/{name}": value for name, value in dataset_losses.get("frames", {}).items()},
                        "learning_rate": scheduler.get_last_lr()[0],
                        **{f"memory_gb/{name}": value for name, value in mem.items()},
                        **{f"perf/{name}": value for name, value in perf.items()}}, iter, images=images)
            for head, by_dataset in dataset_losses.items():
                print(f"val {'action' if head == cfg.model.action_head else head} loss by dataset: "
                      + ", ".join(f"{name} {value:.4f}" for name, value in by_dataset.items()))
            if cfg.model.type == "fm":
                print("Q: " + decode_txt(batch[0][0].tolist())) ## Print the question
                with autocast():
                    answer = model.generate(batch[0][:1], batch[1][:1], batch[4][:1], max_new_tokens=cfg.max_block_size, cfg=cfg)
                print("A: " + decode_txt(answer[0].tolist())) ## Decode the responses learned so far
            util, data_time, sample_time, t_log, iter_log = [], 0.0, 0.0, time.time(), iter ## the eval above isn't counted as training time
            swaps_log = robot_caches["train"].swaps if "train" in robot_caches else 0

        if cfg.sim.eval_interval > 0 and iter > 0 and (iter % cfg.sim.eval_interval == 0 or iter == cfg.max_iters - 1):
            if background_sim is None: ## in this process: training waits
                t = time.time()
                with autocast():
                    out = eval_sim_suites(mini_fm.FMPolicy(model, cfg), cfg, iter)
                save_if_best(iter, log_sim_eval(iter, out, time.time() - t, logger))
                util, data_time, sample_time, t_log, iter_log = [], 0.0, 0.0, time.time(), iter ## the sim eval isn't counted as training time
                swaps_log = robot_caches["train"].swaps if "train" in robot_caches else 0
            else: ## hand a copy of the weights to the background evaluator and keep training
                background_sim.submit(iter, model, wait=iter == cfg.max_iters - 1) ## the final eval always runs
        if background_sim is not None: ## results of earlier background evals
            for result in background_sim.log_finished(logger):
                save_if_best(*result)

        if iter == cfg.max_iters - 1:
            path_ = f"./{model_name}.pth"
            torch.save(model, path_)
            print("Model saved to " + path_)

        # evaluate the loss
        with autocast():
            logits, loss, _ = fast_model(*augment(batch, cfg))
        scaler.scale(loss).backward()

        if (iter + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if hasattr(model, "update_target"): ## JEPA's target encoder follows its encoder, TD-MPC2's target Q-functions its Q-functions, Dreamer's slow critic its critic
                model.update_target()

        if device == 'cuda' and torch.version.hip is None and iter % 10 == 0: ## NVML only: AMD's APUs report no utilization
            util.append(torch.cuda.utilization(device))
        t = time.time()
        batch = next(train_loader)
        sample_time += time.time() - t
        batch = to_device(batch, cfg)
        data_time += time.time() - t

    if background_sim is not None:
        for result in background_sim.close(logger):
            save_if_best(*result)
    for cache in robot_caches.values():
        cache.close()
    logger.close()

    return losses['val']['loss']

if __name__ == "__main__":
    results = my_main()
    print("results:", results)
    ## Exit without the C++ teardown, which can abort once a sim has rendered in this process (sim evals in-process).
    ## Everything is saved and the logger closed by now.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

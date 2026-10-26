## The LIBERO sim: batched rollouts of a policy, for the sim evals during training (mini_train.py) and RL fine-tuning
## (mini_fm_rl.py), and the evals' logging, optionally in a background process so training doesn't wait.
import functools
import os
import numpy as np
import torch


def write_libero_config():
    """ LIBERO asks for its data paths with input() the first time it is imported, which would hang a notebook or a
    job: write its default config (~/.libero/config.yaml, or $LIBERO_CONFIG_PATH) first, if there is none. """
    import importlib.util, yaml
    folder = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    if os.path.exists(os.path.join(folder, "config.yaml")):
        return
    root = os.path.join(importlib.util.find_spec("libero").submodule_search_locations[0], "libero") ## without importing it
    os.makedirs(folder, exist_ok=True)
    paths = {"benchmark_root": root, "bddl_files": os.path.join(root, "bddl_files"), "init_states": os.path.join(root, "init_files"),
             "datasets": os.path.join(root, "..", "datasets"), "assets": os.path.join(root, "assets")}
    with open(os.path.join(folder, "config.yaml"), "w") as f:
        yaml.dump(paths, f)


write_libero_config()


def goal_distances(env):
    """ Distances (m) between the end effector, the object and its goal for the BDDL goal predicates, e.g. (On bowl plate):
    the terms LIBERO's DenseRewardEnv shapes its reward with. Unary goals (Open, Close, Turnon, ...) have no separate
    target, so their object is the goal and they give no object_to_goal. Averaged over the goal's predicates. """
    e = env.env ## the robosuite env under LIBERO's OffScreenRenderEnv
    pos = lambda name: e.object_states_dict[name].get_geom_state()["pos"]
    ee = e.sim.data.site_xpos[e.robots[0].eef_site_id]
    dists = {"ee_to_object": [], "ee_to_goal": [], "object_to_goal": []}
    for state in e.parsed_problem["goal_state"]:
        obj, goal = pos(state[1]), pos(state[-1])
        dists["ee_to_object"].append(np.linalg.norm(ee - obj))
        dists["ee_to_goal"].append(np.linalg.norm(ee - goal))
        if len(state) == 3:
            dists["object_to_goal"].append(np.linalg.norm(obj - goal))
    return {k: float(np.mean(v)) for k, v in dists.items() if v}

## ---- LIBERO: batched rollouts, for the sim evals (eval_libero) and RL fine-tuning (mini_fm_rl.py) ----

LIBERO_NOOP = np.array([0, 0, 0, 0, 0, 0, -1.0]) ## no motion, gripper open


class LiberoChunkEnv:
    """ A LIBERO env (OffScreenRenderEnv) of one task, for rollout_libero's pool of worker processes.
    - reset() is soft: it keeps the MuJoCo sim instead of rebuilding it (34 ms instead of about 1 s). After
      set_init_state an episode matches a fresh env's exactly, once the gripper's smoothed command (which a soft reset
      keeps from the last episode) is zeroed as a fresh env has it.
    - step(actions) runs a whole action chunk in the worker, stopping early if the task succeeds, so each env can step
      its chunk without waiting for the others. step({"init_state": s}) instead starts a new episode from init state s,
      so resets run asynchronously too. The info has the agent-view frames (for videos), the end effector, object and
      goal distances at the end of the chunk (goal_distances) and the seconds it took.
    Everything else (set_init_state, close, ...) is the LIBERO env's. """
    def __init__(self, task, camera_size):
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        self.env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                                      camera_heights=camera_size, camera_widths=camera_size, hard_reset=False)
        self.env.seed(0)

    def reset(self):
        obs = self.env.reset()
        self.env.env.robots[0].gripper.current_action[:] = 0
        return obs

    def step(self, actions):
        import time
        t0, frames = time.time(), []
        if isinstance(actions, dict):
            self.reset()
            return self.env.set_init_state(actions["init_state"]), 0.0, False, {"frames": frames, "seconds": time.time() - t0}
        for action in actions:
            obs, reward, done, info = self.env.step(action)
            frames.append(obs["agentview_image"][::-1])
            if done: ## LIBERO ends an episode only when its task succeeds
                break
        return obs, reward, done, {"frames": frames, "distances": goal_distances(self.env), "seconds": time.time() - t0}

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_libero_pool(task_suite, tasks, envs_per_task, camera_size, wait_num):
    """ envs_per_task parallel LIBERO envs (LIBERO's SubprocVectorEnv, one process each) for each task id in tasks, to
    keep open for all of a run's rollout_libero calls. They step asynchronously: envs.step returns as soon as wait_num
    of the envs it is waiting for have finished their chunk. Returns (envs, the task id of each env). """
    import multiprocessing
    from libero.libero.envs import SubprocVectorEnv
    env_tasks = [t for t in tasks for _ in range(envs_per_task)]
    ## The workers start from a clean server process (forkserver), not as forks of this one: a fork can't render
    ## (MuJoCo's EGL) once this process holds other GPU contexts. The server imports LIBERO once, so the workers share
    ## that memory (about 1 GB each otherwise).
    start_method = multiprocessing.get_start_method()
    multiprocessing.set_start_method("forkserver", force=True)
    multiprocessing.set_forkserver_preload(["__main__", "libero.libero.envs"])
    envs = SubprocVectorEnv([functools.partial(LiberoChunkEnv, task_suite.get_task(t), camera_size) for t in env_tasks],
                            wait_num=min(wait_num, len(env_tasks)))
    multiprocessing.set_start_method(start_method, force=True)
    return envs, env_tasks


def rollout_libero(policy, task_suite, envs, env_tasks, jobs, episode_length, wait_steps, noise_std=0.0,
                   init_states=None, video_jobs=(0,), goal_images=None):
    """ One episode for each (task id, init state id) of task_suite in jobs, on a pool of envs from make_libero_pool:
    env e only runs task env_tasks[e], and when its episode ends it starts that task's next job. The envs run
    asynchronously: whenever some of them finish their chunk (or reset), the policy is queried for those (in one batch)
    and they start their next chunk, while the others keep simulating. The policy is an FMPolicy (mini_fm.py): it sees
    the camera cfg.sim.obs_key (flipped vertically, as LIBERO renders it upside down), the goal text, the robot pose (if
    cfg.policy.use_pose_data) and a goal image: goal_images[job] if given (dataset_goal_images, for the planners),
    else the episode's first frame. Each policy query predicts a whitened action chunk; Gaussian noise of std noise_std
    is added (the exploration of RL fine-tuning, 0 for evals) and the whole chunk runs. An episode is wait_steps no-op
    steps (rounded up to whole chunks) so the objects settle, then up to episode_length policy steps. It starts from
    init_states[task id][init state id] (LIBERO's fixed benchmark init states for the tasks it doesn't give).
    Returns (success of each job, {"obs", "txt", "pose", "action", "episode"}: the policy inputs and whitened action of
    every query and the job it belongs to, {job: its agent-view frames} for the jobs in video_jobs, timing: the seconds
    this process spent on policy queries (policy_seconds, of which model_seconds in the model itself; the rest prepares
    its inputs and decodes its actions) and waiting for envs, and the seconds the envs spent simulating and resetting,
    and the goal_distances of each job at the end of its episode). """
    import time
    cfg = policy._cfg
    chunk = cfg.policy.action_stacking
    wait = -(-wait_steps // chunk) * chunk
    todo = {t: [j for j, (tj, _) in enumerate(jobs) if tj == t] for t in set(env_tasks)} ## jobs not started yet, per task
    assert sum(map(len, todo.values())) == len(jobs), "every job's task needs an env in the pool"
    txt = {t: policy.encode_text_goal(task_suite.get_task(t).language) for t in todo}
    init_states = {t: task_suite.get_task_init_states(t) for t in todo} | (init_states or {})
    job = np.full(len(env_tasks), -1) ## the job each env runs, -1 when its task has none left
    steps = np.zeros(len(env_tasks), int) ## env steps into its episode, the no-op ones included
    obs, goal_image = [None] * len(env_tasks), [None] * len(env_tasks)
    success, distances = np.zeros(len(jobs), bool), [None] * len(jobs)
    queries, videos = {"obs": [], "txt": [], "pose": [], "action": [], "episode": []}, {j: [] for j in video_jobs}
    obs_keys = cfg.sim.get("obs_key", "agentview_image")
    obs_keys = [obs_keys] if isinstance(obs_keys, str) else list(obs_keys) ## dataset.cameras: several, side by side
    camera = lambda e: obs[e][obs_keys[0]][::-1]
    observe = lambda e: np.concatenate([policy.preprocess_state(obs[e][k][::-1]) for k in obs_keys], axis=1) ## the policy's input
    timing = {"policy_seconds": 0.0, "model_seconds": 0.0, "env_wait_seconds": 0.0, "sim_seconds": 0.0, "policy_calls": 0}

    def reset_message(e): ## the message that starts env e on its task's next job, or None when there is none left
        job[e] = todo[env_tasks[e]].pop(0) if todo[env_tasks[e]] else -1
        steps[e] = 0
        return {"init_state": init_states[jobs[job[e]][0]][jobs[job[e]][1]]} if job[e] >= 0 else None

    messages = {e: reset_message(e) for e in range(len(env_tasks))} ## what to send each env next
    messages = {e: m for e, m in messages.items() if m is not None}
    resetting = np.zeros(len(env_tasks), bool)
    ready, n_stepping = [], 0 ## envs waiting for a policy query, and envs simulating a chunk or a reset
    while messages or ready or n_stepping:
        t0 = time.time()
        if ready:
            ids = np.array(ready)
            actions = np.tile(LIBERO_NOOP, (len(ids), chunk, 1))
            acting = steps[ids] >= wait ## the others are still settling, on no-ops
            if acting.any():
                q = ids[acting]
                image = torch.tensor(np.stack([observe(e) for e in q]), dtype=torch.float32, device=cfg.device)
                goal_img = torch.tensor(np.stack([goal_image[e] for e in q]), dtype=torch.float32, device=cfg.device)
                goal = torch.cat([txt[env_tasks[e]] for e in q])
                pose = (policy.encode_pose(torch.tensor(np.stack([policy.robot_pose(obs[e]) for e in q]), dtype=torch.float32, device=cfg.device))
                        if cfg.policy.use_pose_data else None)
                t_model = time.time()
                with torch.no_grad():
                    action = policy(image, goal, goal_img, pose=pose)[0]
                if action.is_cuda: ## wait for the GPU, so model_seconds is the inference, not only its launch
                    torch.cuda.synchronize()
                timing["model_seconds"] += time.time() - t_model
                action = action + noise_std * torch.randn_like(action)
                for k, v in zip(queries, (image, goal, pose, action, torch.from_numpy(job[q]))):
                    queries[k].append(v)
                actions[acting] = policy.decode_action(action).cpu().numpy().reshape(len(q), chunk, -1)
                timing["policy_calls"] += 1
            messages.update(zip(ids.tolist(), actions))
            ready = []
        t1 = time.time()
        timing["policy_seconds"] += t1 - t0
        ## send the messages, and get back whichever envs finish theirs first
        ids = list(messages)
        resetting[[e for e in ids if isinstance(messages[e], dict)]] = True
        step_obs, _, done, info = envs.step([messages[e] for e in ids], id=ids) if ids else envs.step(None)
        timing["env_wait_seconds"] += time.time() - t1
        n_stepping += len(ids) - len(info)
        messages = {}
        for o, d, i in zip(step_obs, done, info):
            e = i["env_id"]
            obs[e] = o
            timing["sim_seconds"] += i["seconds"]
            if resetting[e]: ## a new episode started
                resetting[e] = False
                goal_image[e] = goal_images[job[e]] if job[e] in (goal_images or {}) else policy.preprocess_state(camera(e))
                ready.append(e)
                continue
            steps[e] += chunk
            if job[e] in videos and steps[e] > wait: ## the policy's steps, not the settling no-ops
                videos[job[e]] += i["frames"]
            success[job[e]] |= d
            if d or steps[e] >= wait + episode_length:
                distances[job[e]] = i["distances"]
                m = reset_message(e)
                if m is not None:
                    messages[e] = m
            else:
                ready.append(e)
    queries = {k: (torch.cat(v) if v and v[0] is not None else None) for k, v in queries.items()}
    return success, queries, videos, timing, distances


def eval_libero(policy, cfg, iter_=0, log_dir="./"):
    """ Runs the policy in LIBERO suite cfg.sim.task_set: one episode for each task id of cfg.sim.eval_tasks and init
    state id of cfg.sim.num_init_states, on a pool of one env per task opened for this call (rollout_libero: the tasks
    run in parallel, each env about 1.5 GB of RAM, and the init states of a task one after another). The init
    states are those the demos of the dataset trained on the suite started from (dataset_init_states; LIBERO's fixed
    benchmark ones for tasks without demos). A planner (a policy whose model is goal_conditioned: mini_jepa.py's CEM)
    gets the last frame of each episode's demo as its goal image (dataset_goal_images). Saves a video of every episode
    to log_dir. Returns {"success_rate", "rewards" (LIBERO's reward is 1 on success, so the same), "num_episodes", the
    mean end distances (goal_distances), "per_task": {task id: the same stats over its episodes}, "policy_seconds" and
    "sim_seconds": the time spent querying the policy (of which "model_seconds" in the model itself) and simulating}. """
    import imageio, logging
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[cfg.sim.task_set]()
    tasks, init_ids = list(cfg.sim.eval_tasks), list(cfg.sim.num_init_states)
    for t in tasks:
        print(f"[info] retrieving task {t} from suite {cfg.sim.task_set}: {suite.get_task(t).language}")
    jobs = [(t, i) for t in tasks for i in init_ids]
    init_states = dataset_init_states(cfg, suite, cfg.sim.task_set)
    print(f"[info] init states of the dataset's demos for {len(set(tasks) & set(init_states))} of {len(tasks)} tasks, LIBERO's for the rest")
    goal_images = None
    if getattr(getattr(policy, "model", None), "goal_conditioned", False): ## a planner: the last frame of each episode's demo is its goal
        goal_images = dataset_goal_images(cfg, suite, cfg.sim.task_set, jobs)
        logging.getLogger(__name__).info(f"goal images (the demos' last frames) for {len(goal_images)} of {len(jobs)} episodes")
    was_training = policy.training
    policy.eval()
    envs, env_tasks = make_libero_pool(suite, tasks, 1, cfg.sim.get("camera_size", 256), wait_num=1)
    success, _, videos, timing, distances = rollout_libero(policy, suite, envs, env_tasks, jobs, cfg.sim.episode_length,
                                                           cfg.sim.get("wait_steps", 0), init_states=init_states,
                                                           video_jobs=range(len(jobs)), goal_images=goal_images)
    envs.close()
    policy.train(was_training)
    paths = [os.path.join(log_dir, f"libero-{iter_}-task-id-{t}-init-id-{i}.mp4") for t, i in jobs]
    for j, path in enumerate(paths):
        imageio.mimsave(path, videos[j], fps=20)

    def stats(js): ## over the episodes of jobs js
        out = {"success_rate": float(success[js].mean()), "rewards": float(success[js].mean()), "num_episodes": len(js)}
        out.update({k: float(np.mean([distances[j][k] for j in js if k in distances[j]]))
                    for k in ("ee_to_object", "ee_to_goal", "object_to_goal") if any(k in distances[j] for j in js)})
        return out
    return {**stats(list(range(len(jobs)))),
            "per_task": {t: stats([j for j, (tj, _) in enumerate(jobs) if tj == t]) for t in tasks},
            ## the seconds spent querying the policy and waiting for the envs to simulate
            "policy_seconds": timing["policy_seconds"], "model_seconds": timing["model_seconds"], "sim_seconds": timing["env_wait_seconds"]}


def task_set_of(repo_id):
    """ The LIBERO suite a robot dataset on the Hub is evaluated in: gberseth/libero_90_no_noops_lerobot -> libero_90. """
    return repo_id.split("/")[-1].split("_no_noops")[0]


def action_stats(cfg, task_set):
    """ (mean, std) to un-whiten the policy's actions with in a task set: those of the dataset trained on its tasks
    (dataset.action_stats), or for a task set no dataset was trained on, those of all the robot data
    (dataset.action_mean/std). """
    for repo_id, stats in (cfg.dataset.get("action_stats") or {}).items():
        if task_set_of(repo_id) == task_set:
            return stats["mean"], stats["std"]
    return cfg.dataset.action_mean, cfg.dataset.action_std


## LIBERO's standard max steps per suite (from openvla's run_libero_eval.py), unless sim.simulators.libero.episode_length is set
LIBERO_MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
## LIBERO's name for each dataset camera
LIBERO_CAMERAS = {"observation.images.wrist_image": "robot0_eye_in_hand_image", "observation.images.image": "agentview_image"}


def image_key_of(cfg, task_set):
    """ The camera (dataset image key) the model was trained on for a task set: that of the dataset of its tasks
    (dataset.robot_image_key, one per dataset), or for a task set no dataset was trained on, the first dataset's. With
    dataset.cameras it is a list of cameras, side by side. """
    keys, names = cfg.dataset.robot_image_key, cfg.dataset.robot_dataset_name
    if isinstance(keys, str):
        return keys
    names = [names] if isinstance(names, str) else list(names)
    return next((k for n, k in zip(names, keys) if task_set_of(n) == task_set), keys[0])


@functools.cache ## once per process: the background evaluator reads it at every eval
def libero_demos(repo_id):
    """ {demo HDF5 file: (the dataset's episode index of each of its demos, the MuJoCo states they started from,
    (demos, state size)), in episode order} of a LIBERO dataset on the Hub, from its meta/libero_init_states.parquet. """
    import pandas as pd
    from huggingface_hub import hf_hub_download
    demos = pd.read_parquet(hf_hub_download(repo_id, "meta/libero_init_states.parquet", repo_type="dataset"))
    return {file: (rows.episode_index.to_numpy(), np.stack(rows.init_state.to_numpy()))
            for file, rows in demos.groupby(demos.demo.str.split("/").str[0])}


def libero_repo_of(cfg, suite):
    """ The dataset trained on a LIBERO suite's tasks, or None. """
    names = cfg.dataset.get("robot_dataset_name") or []
    names = [names] if isinstance(names, str) else list(names)
    return next((n for n in names if task_set_of(n) == suite), None)


def dataset_init_states(cfg, task_suite, suite):
    """ {task id: its init states} for a LIBERO suite's evals: the states the demos of the dataset trained on its tasks
    started from, so init state i of a task sets the scene up as its i-th demo was. LIBERO's own benchmark init states
    are other ones. Tasks are matched by their LIBERO name (their HDF5 file), not their goal text, which several
    libero_90 scenes share. Tasks without demos (libero_90's LIVING_ROOM_SCENE2 butter task) and suites no dataset was
    trained on are left out, so they keep LIBERO's. """
    repo_id = libero_repo_of(cfg, suite)
    demos = libero_demos(repo_id) if repo_id else {}
    files = {t: task_suite.get_task(t).name + "_demo.hdf5" for t in range(task_suite.n_tasks)}
    return {t: demos[f][1] for t, f in files.items() if f in demos}


def dataset_goal_images(cfg, task_suite, suite, jobs):
    """ {job: its goal image} for jobs, (task id, init state id) of a LIBERO suite, that start from the dataset's demos
    (dataset_init_states): the last frame of the demo that started from that init state, at the training image size,
    scaled as the policy's inputs, (H, W, 3). For the policies that plan towards a goal image (mini_jepa.py's CEM,
    mini_vlaps.py); jobs of tasks without demos get none. """
    from data_loading import stream_last_frames, scale_image
    repo_id = libero_repo_of(cfg, suite)
    demos = libero_demos(repo_id) if repo_id else {}
    files = [task_suite.get_task(t).name + "_demo.hdf5" for t, _ in jobs]
    found = [j for j, f in enumerate(files) if f in demos]
    if not found:
        return {}
    key = image_key_of(cfg, suite) ## the camera trained on, or with dataset.cameras a list
    frames = stream_last_frames(repo_id, [key] if isinstance(key, str) else list(key),
                                [demos[files[j]][0][jobs[j][1]] for j in found], (cfg.image_shape[1], cfg.image_shape[0]))
    return {j: scale_image(frame.astype(np.float32), cfg.image_bounds) for j, frame in zip(found, frames)}


## ---- Sim evals during training (mini_train.py) ----
## The trainer hands over a policy (an FMPolicy that wraps its model); these run it in every LIBERO suite of
## sim.task_sets, log the results (to a Logger, logger.py), and can do it in a background process so training doesn't wait.

def eval_task_sets(cfg):
    """ The LIBERO suites the sim evals run: every one listed under sim.simulators.libero.task_sets, or sim.task_sets
    when it is given (as the tests do). They don't depend on the datasets trained on. """
    return list(cfg.sim.task_sets or cfg.sim.simulators.libero.get("task_sets") or [])


def eval_sim_suites(policy, cfg, iter):
    """ Runs the policy in each LIBERO suite of sim.task_sets (eval_libero). Returns {suite: eval_libero's result, with
    the suite's seconds}. Videos go to sim_eval/<suite>/ in the run folder. """
    import random, time
    from libero.libero import benchmark
    from omegaconf import open_dict
    results = {}
    libero = cfg.sim.simulators.libero ## how much each eval runs (eval_libero reads it from cfg.sim)
    for suite in cfg.sim.task_sets:
        policy.task_set = suite ## its actions are un-whitened with the stats of the dataset of this suite (action_stats)
        log_dir = os.path.join("sim_eval", suite)
        os.makedirs(log_dir, exist_ok=True)
        start = time.time()
        ## A random sample of tasks_per_eval tasks (a new one at each step, so all get covered over training)
        tasks = list(libero.eval_tasks) if libero.eval_tasks is not None else list(range(benchmark.get_benchmark_dict()[suite]().n_tasks))
        tasks = sorted(random.Random(f"{suite}-{iter}").sample(tasks, min(libero.tasks_per_eval, len(tasks))))
        with open_dict(cfg):
            cfg.sim.task_set = suite
            key = image_key_of(cfg, suite) ## the camera trained on for this suite, or with dataset.cameras a list
            cfg.sim.obs_key = LIBERO_CAMERAS[key] if isinstance(key, str) else [LIBERO_CAMERAS[k] for k in key]
            cfg.sim.eval_tasks, cfg.sim.num_init_states = tasks, libero.num_init_states
            cfg.sim.episode_length, cfg.sim.wait_steps = libero.episode_length or LIBERO_MAX_STEPS[suite], libero.wait_steps
        results[suite] = {**eval_libero(policy, cfg, iter, log_dir), "seconds": time.time() - start}
    return results


def sim_videos(iter, suite):
    """ {name: path} of the rollout videos eval_libero saved for this step, e.g. task_3_init_0. """
    import glob, re
    paths = sorted(glob.glob(os.path.join("sim_eval", suite, f"*-{iter}-task-id-*-init-id-*.mp4")))
    return {"task_{}_init_{}".format(*re.search(r"task-id-(\d+)-init-id-(\d+)", p).groups()): p for p in paths}


def log_sim_eval(iter, results, seconds, logger):
    """ Logs one sim eval (to the run folder's mini_train.log too); returns its overall success: the mean over its
    suites. """
    import logging
    log = logging.getLogger(__name__)
    ## where the eval's time went: querying the policy, simulating, and the rest (making the envs, saving videos)
    policy_s, sim_s = (sum(r.get(k, 0.0) for r in results.values()) for k in ("policy_seconds", "sim_seconds"))
    log.info(f"step {iter}: sim success rate " + ", ".join(f"{suite} {r['success_rate']:.2f} (tasks {sorted(r['per_task'])})"
                                                         for suite, r in results.items())
          + f" ({seconds:.0f}s: policy {policy_s:.0f}s, simulation {sim_s:.0f}s)")
    ## success of each suite (over its sampled tasks) and their mean, the overall success
    scalars = {"sim/eval_seconds": seconds, "sim/policy_seconds": policy_s, "sim/simulation_seconds": sim_s,
               "sim/model_seconds": sum(r.get("model_seconds", 0.0) for r in results.values()), ## the model's part of policy_seconds
               **{f"sim_seconds/{suite}": r["seconds"] for suite, r in results.items()},
               "sim/success_rate": float(np.mean([r["success_rate"] for r in results.values()]))}
    per_task = {} ## TensorBoard only: in wandb they would bury the per-suite curves
    videos = {} ## rollout MP4s
    for suite, r in results.items():
        scalars[f"sim/success_rate_{suite}"] = r["success_rate"]
        ## end-of-episode distances (m) between the end effector, the object and its goal (goal_distances)
        scalars.update({f"sim_distance/{k}_{suite}": r[k] for k in ("ee_to_object", "ee_to_goal", "object_to_goal") if k in r})
        per_task.update({f"sim_{suite}/task_{task_id}": stats["success_rate"] for task_id, stats in r.get("per_task", {}).items()})
        videos.update({f"sim_video_{suite}/{name}": path for name, path in sim_videos(iter, suite).items()})
    logger.log(scalars, iter, tensorboard_only=per_task, videos=videos)
    return scalars["sim/success_rate"]


class BackgroundSimEval:
    """ Runs the sim evals in a separate process at the lowest OS priority, so training never waits for them.
    make_policy(cfg) builds the policy there (it must be a top-level function, to reach the process); submit() hands it
    a copy of the model's weights, which it loads into policy.model. The job queue holds one snapshot, so evals never
    pile up: a step whose eval would have to wait is skipped. The weights of each submitted step are also kept here
    until its result is back, so the trainer can save them if they are the best so far. If the evaluator crashes (e.g.
    MuJoCo's EGL renderer now and then fails to make a framebuffer), the next submit() starts a new one, so one crash
    loses one eval, not all the later ones. """

    def __init__(self, make_policy, cfg):
        from omegaconf import OmegaConf
        self.make_policy, self.cfg = make_policy, OmegaConf.to_container(cfg, resolve=True)
        self.weights = {} ## step -> the weights its eval runs on, until the eval is done
        self._start()

    def _start(self):
        import atexit, multiprocessing as mp
        ctx = mp.get_context("spawn") ## not fork: the training process already uses CUDA
        self.jobs, self.results = ctx.Queue(maxsize=1), ctx.Queue()
        ## Not a daemon: eval_libero starts its own pool of env processes, which daemons can't. So it isn't killed
        ## automatically when training exits; close() stops it normally, and atexit stops it if training crashes.
        self.process = ctx.Process(target=_background_sim_eval, args=(self.make_policy, self.cfg, self.jobs, self.results))
        self.process.start()
        atexit.register(self.process.terminate)

    def submit(self, iter, model, wait=False):
        import queue
        weights = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        while True: ## waiting (the final eval) polls, so it notices if the busy evaluator crashes
            if not self.process.is_alive(): ## it crashed (its traceback is in the log): its eval is lost, start a new one
                print(f"step {iter}: the sim evaluator process died (lost the eval of step {sorted(self.weights)}), restarting it")
                self.weights = {}
                self._start()
            try:
                self.jobs.put((iter, weights), block=wait, timeout=5)
                self.weights[iter] = weights
                return
            except queue.Full:
                if not wait:
                    print(f"step {iter}: sim eval skipped, the evaluator is still busy with the previous one")
                    return

    def _log(self, result, logger): ## (step, overall success, the step's weights)
        iter, out, seconds = result
        return iter, log_sim_eval(iter, out, seconds, logger), self.weights.pop(iter)

    def log_finished(self, logger):
        """ Logs the evals finished so far; returns [(step, overall success, the step's weights)]. """
        finished = []
        while not self.results.empty():
            finished.append(self._log(self.results.get(), logger))
        return finished

    def close(self, logger):
        """ Waits for the evals still running, including the final one, and logs them; returns them as log_finished does. """
        import queue
        if self.process.is_alive():
            self.jobs.put(None)
        finished = []
        while self.process.is_alive() or not self.results.empty():
            try:
                finished.append(self._log(self.results.get(timeout=5), logger))
            except queue.Empty:
                pass
        return finished


def _background_sim_eval(make_policy, cfg, jobs, results):
    """ The BackgroundSimEval process: takes (step, weights) from jobs, evaluates them, puts (step, results, seconds)
    on results; stops at None. """
    import sys, time
    from omegaconf import OmegaConf
    os.nice(19)
    os.environ.setdefault("MUJOCO_GL", "egl")
    cfg = OmegaConf.create(cfg)
    cfg.device = cfg.sim.device ## the CPU by default, so the evaluator doesn't take GPU time from training
    if cfg.device == "cpu":
        torch.set_num_threads(1)
    policy = make_policy(cfg).to(cfg.device)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.precision)
    while (job := jobs.get()) is not None:
        iter, weights = job
        policy.model.load_state_dict(weights)
        t = time.time()
        try:
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None and cfg.device == "cuda"):
                out = eval_sim_suites(policy, cfg, iter)
        except BaseException:
            ## e.g. an env process crashed. Exit at once: a normal exit can hang waiting on the broken env pool, and
            ## then the trainer, which restarts a dead evaluator at its next eval, would wait on it until the time limit.
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        results.put((iter, out, time.time() - t))
    ## Exit without the C++ teardown (which can abort in a process that has rendered with EGL); the results are
    ## flushed to the trainer first
    results.close()
    results.join_thread()
    os._exit(0)

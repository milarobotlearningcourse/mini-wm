## Data loading for the robot foundation models (used by mini_train.py): LeRobot robot trajectories (goal
## text + camera images -> action chunk), turned into batches of (X_text, X_img, Y, A, P), and a cache that keeps the
## datasets on the GPU so batches cost no CPU decoding.
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader


def build_vocab(chars):
    chars = list(dict.fromkeys(chars)) ## de-duplicate, keep order
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    unk = stoi[' '] ## Unknown characters are mapped to a space
    encode_txt = lambda s: [stoi.get(c, unk) for c in s]
    decode_txt = lambda l: ''.join([itos[i] for i in l])
    return chars, encode_txt, decode_txt

## Every loader yields (X_text, X_img, Y, A, P): Y is the next-token target (cross entropy loss), A the action target
## (MSE loss) and P the robot's pose (POSE_KEY: the arm's joint positions), whitened. Actions are whitened with the stats
## of their own dataset (dataset.action_stats), since their units differ (e.g. LIBERO's are in [-1, 1], Bridge's in m/s
## and rad/s: VELOCITY_DATASETS), after their rotations are made axis-angle (to_batch_actions); poses with those of all
## the robot data (dataset.pose_mean/std).
## A is an action chunk: the frame's action and the next policy.action_stacking - 1 of its trajectory (the last action
## repeated past its end), whitened and flattened to action_stacking * action_dim.
## Rows with Y = -1 are robot rows scored only on A; all other rows are scored only on Y and have A = 0 and P = 0.

## The robot's pose the models take as input: the 7 arm joint positions. The actions they predict are end effector
## motions, so the input is kept in a different space from the output (with the end effector pose as input, the model
## can predict the next action from its own pose within a demo, which doesn't carry over to closed-loop control).
POSE_KEY = "observation.joint_state"

## The control frequency of the action chunks the models predict (LIBERO's and ManiSkill's). A dataset recorded slower
## (Bridge, 5 Hz) has each action held for CONTROL_HZ / fps steps of its chunks (a, a, a, a, b, ... for Bridge), so
## a chunk always spans the same time (action_stacking / CONTROL_HZ seconds). Bridge's held value is a velocity
## (VELOCITY_DATASETS); a sim at the dataset's rate runs every (CONTROL_HZ / fps)-th action of a chunk.
CONTROL_HZ = 20


def hold_steps(fps):
    """ Steps of a CONTROL_HZ chunk that each action of an fps dataset is held for. """
    assert CONTROL_HZ % fps == 0, f"{fps} Hz data doesn't divide the {CONTROL_HZ} Hz control rate"
    return CONTROL_HZ // int(fps)


def to_rgb(img, size):
    """ A PIL image or an (H, W, 3) array as an RGB uint8 array of the given (width, height). """
    import cv2
    if hasattr(img, "mode") and img.mode != "RGB":
        img = img.convert("RGB")
    img = np.asarray(img, dtype=np.uint8)
    if img.ndim == 2: ## grayscale, from a numpy-formatted image column
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img[..., :3] ## drop alpha
    return img if img.shape[:2] == size[::-1] else cv2.resize(img, size)


## Demos recorded in ManiSkill's Bridge twins (the WidowX), whose actions are in the twin controller's units: the delta
## rotation as XYZ Euler angles and the gripper as a finger position target (0.014 closed - 0.038 open). They are
## converted to Bridge's labels (delta axis-angle, gripper -1 open / 1 close), so all the WidowX data, real and sim,
## agrees, and a model's actions mean the same on the real robot as in the twins.
BRIDGE_TWIN_DEMOS = ["gberseth/maniskill_putspoonontablecloth_lerobot", "gberseth/maniskill_putcarrotonplate_lerobot",
                     "gberseth/maniskill_stackcube_lerobot"]


def to_bridge_actions(actions):
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_euler("XYZ", actions[:, 3:6]).as_rotvec()
    open_ = 2 * (actions[:, 6:] - 0.014) / (0.038 - 0.014) - 1 ## the twin's gripper controller maps [-1, 1] to 0.014 - 0.038
    return np.concatenate([actions[:, :3], rot, -open_], axis=1).astype(np.float32)


## Bridge's actions, and those of the demos in its twins (in Bridge's labels), are end-effector velocities in the
## batches (m/s and rad/s: each 5 Hz step's motion times fps), so a value held for several steps of a CONTROL_HZ chunk
## means the same motion in each. The eval in the twins integrates them over the sim's control step
## (sim_eval.eval_maniskill).
VELOCITY_DATASETS = ["gberseth/bridge_orig_lerobot", *BRIDGE_TWIN_DEMOS]

## The ManiSkill datasets' actions were recorded for its pd_ee_delta_pose controller, which reads the delta rotation as
## XYZ Euler angles (though the datasets name those columns axis_angle); the other datasets' are axis-angle. They are
## axis-angle (in rad) in the batches, so a rotation means the same in every dataset; the ManiSkill evals convert back
## (sim_eval.eval_maniskill). The stored rotation is the controller's normalized input: it scales it to a length of at
## most 1, then multiplies it by its rot_lower, MANISKILL_ROT_SCALE (the Panda's and the stick's), to get the angles in rad.
EULER_DATASETS = ["gberseth/maniskill_panda_lerobot", "gberseth/maniskill_panda_stick_lerobot", "gberseth/maniskill_pusht_lerobot"]
MANISKILL_ROT_SCALE = -0.1


def to_batch_actions(actions, repo_id, fps):
    """ A dataset's stored actions, (n, action_dim), as the batches hold them: the delta rotation as axis-angle
    (EULER_DATASETS) and the motion as a velocity (VELOCITY_DATASETS: the six pose dimensions times fps). """
    from scipy.spatial.transform import Rotation
    if repo_id in EULER_DATASETS: ## the Euler angles the controller executed, in rad, as axis-angle
        rot = actions[:, 3:6] / actions[:, 3:6].norm(dim=1, keepdim=True).clamp(min=1) * MANISKILL_ROT_SCALE
        rot = torch.from_numpy(Rotation.from_euler("XYZ", rot.numpy()).as_rotvec()).to(actions.dtype)
        actions = torch.cat([actions[:, :3], rot, actions[:, 6:]], dim=1)
    if repo_id in VELOCITY_DATASETS:
        actions = torch.cat([actions[:, :6] * fps, actions[:, 6:]], dim=1)
    return actions


def load_lerobot(repo_id, image_key):
    """ Opens a LeRobot v3 dataset without the lerobot library: frames come from the memory-mapped parquet files,
    images stay in the MP4 files and are decoded per sample. image_key is one camera, or a list of cameras shown side
    by side (dataset.cameras), where a camera the dataset doesn't have is left blank. Each episode gets the MP4 file
    ("video") and start time ("start") of each camera, None for a blank one. Returns (frames, episodes, tasks). """
    import glob, json, datasets, pandas as pd, pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError
    keys = [image_key] if isinstance(image_key, str) else list(image_key)
    ## Only the cameras trained on. One MP4 per episode means about one Hub request per file, and the Hub allows 1000
    ## per 5 minutes. When rate limited, snapshot_download raises a 429 or quietly returns the partial cached snapshot,
    ## so check every episode's video is there, and otherwise wait and resume (files already downloaded are kept).
    for attempt in range(30):
        try:
            root = snapshot_download(repo_id, repo_type="dataset", allow_patterns=["meta/*", "data/*", *(f"videos/{k}/*" for k in keys)])
            info = json.load(open(f"{root}/meta/info.json"))
            present = [k for k in keys if k in info["features"]]
            assert present, f"{repo_id} has no camera {image_key} (e.g. our Bridge export has no wrist camera)"
            episodes = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))])
            episodes = episodes.sort_values("episode_index").reset_index(drop=True)
            videos = {k: [f"{root}/videos/{k}/chunk-{c:03d}/file-{f:03d}.mp4" for c, f in
                          zip(episodes[f"videos/{k}/chunk_index"], episodes[f"videos/{k}/file_index"])] for k in present}
            data_files = sorted(glob.glob(f"{root}/data/*/*.parquet"))
            missing = sum(not os.path.exists(path) for paths in videos.values() for path in set(paths))
            if (missing == 0 and len(episodes) == info["total_episodes"]
                    and sum(pq.ParquetFile(f).metadata.num_rows for f in data_files) == info["total_frames"]):
                break
            print(f"{repo_id}: not all downloaded yet ({missing} videos missing; Hub rate limit?), retrying in 60 s")
        except (FileNotFoundError, ValueError): ## a partial snapshot without some meta or data files
            print(f"{repo_id}: not all downloaded yet (Hub rate limit?), retrying in 60 s")
        except HfHubHTTPError as e:
            if e.response is None or e.response.status_code != 429:
                raise
            print(f"{repo_id}: Hub rate limit, retrying in 60 s")
        time.sleep(60)
    else:
        raise RuntimeError(f"{repo_id}: could not download all the videos")
    frames = datasets.load_dataset("parquet", data_files=data_files, split="train")
    frames = frames.select_columns(["action", POSE_KEY, "timestamp", "task_index", "episode_index"])
    if repo_id in BRIDGE_TWIN_DEMOS:
        actions = to_bridge_actions(np.stack(frames.with_format("numpy")[:]["action"]))
        frames = frames.remove_columns("action").add_column("action", list(actions)).cast_column("action", datasets.Sequence(datasets.Value("float32"), 7))
    if len(present) < len(keys):
        print(f"{repo_id}: no camera {[k for k in keys if k not in present]}, left blank")
    ## Per camera: the episode's MP4 file and where the episode starts in it (a file may hold several episodes)
    episodes["video"] = [[videos[k][i] if k in present else None for k in keys] for i in range(len(episodes))]
    episodes["start"] = [[episodes[f"videos/{k}/from_timestamp"][i] if k in present else None for k in keys] for i in range(len(episodes))]
    episodes["fps"] = info["fps"]
    tasks = pd.read_parquet(f"{root}/meta/tasks.parquet").sort_values("task_index").index.tolist() ## task_index -> goal text
    return frames, episodes[["episode_index", "length", "dataset_from_index", "video", "start", "fps"]], tasks


def stream_lerobot_episode(repo_id, image_keys, index, size):
    """ One episode of a LeRobot v3 dataset, read from the Hub with range requests instead of downloading the dataset
    (LIBERO-90's videos alone are 530 MB): for a quick look at the data, as in the tutorial notebook. Returns (frames,
    video, episodes, tasks): the episode's frames (a DataFrame), its cameras' images side by side (length, H, W *
    cameras, 3), the metadata of every episode, and the goal text of each task index. """
    import json, pandas as pd
    from huggingface_hub import HfFileSystem
    fs, root = HfFileSystem(), f"datasets/{repo_id}"
    info = json.load(fs.open(f"{root}/meta/info.json"))
    video_columns = [f"videos/{k}/{c}" for k in image_keys for c in ("chunk_index", "file_index", "from_timestamp")]
    episodes = pd.concat([pd.read_parquet(f"hf://{f}", columns=["episode_index", "length", *video_columns])
                          for f in sorted(fs.glob(f"{root}/meta/episodes/*/*.parquet"))]).sort_values("episode_index")
    ep = episodes[episodes["episode_index"] == index].iloc[0]
    frames = pd.concat([pd.read_parquet(f"hf://{f}", filters=[("episode_index", "==", index)])
                        for f in sorted(fs.glob(f"{root}/data/*/*.parquet"))])
    ## PyAV seeks in the remote MP4, so only the episode's part of the file is fetched
    mp4 = lambda k: fs.open(f"{root}/videos/{k}/chunk-{int(ep[f'videos/{k}/chunk_index']):03d}/file-{int(ep[f'videos/{k}/file_index']):03d}.mp4")
    length = int(ep["length"])
    video = side_by_side([decode_episode(mp4(k), ep[f"videos/{k}/from_timestamp"], length, info["fps"], size) for k in image_keys],
                         length, size)
    tasks = pd.read_parquet(f"hf://{root}/meta/tasks.parquet").sort_values("task_index").index.tolist()
    return frames, video, episodes.reset_index(drop=True), tasks


def stream_last_frames(repo_id, image_keys, indices, size):
    """ The last frame of each episode in indices of a LeRobot v3 dataset, read from the Hub as stream_lerobot_episode
    does, but decoding only that frame: (len(indices), H, W * cameras, 3) uint8, the cameras side by side. The goal
    images of the planners (sim_eval.dataset_goal_images). """
    import json, pandas as pd
    from huggingface_hub import HfFileSystem
    fs, root = HfFileSystem(), f"datasets/{repo_id}"
    fps = json.load(fs.open(f"{root}/meta/info.json"))["fps"]
    video_columns = [f"videos/{k}/{c}" for k in image_keys for c in ("chunk_index", "file_index", "from_timestamp")]
    episodes = pd.concat([pd.read_parquet(f"hf://{f}", columns=["episode_index", "length", *video_columns])
                          for f in sorted(fs.glob(f"{root}/meta/episodes/*/*.parquet"))]).set_index("episode_index")

    def last_frame(ep, k):
        mp4 = fs.open(f"{root}/videos/{k}/chunk-{int(ep[f'videos/{k}/chunk_index']):03d}/file-{int(ep[f'videos/{k}/file_index']):03d}.mp4")
        return to_rgb(decode_frames(mp4, [ep[f"videos/{k}/from_timestamp"] + (ep["length"] - 1) / fps], fps)[0], size)[None]
    return np.concatenate([side_by_side([last_frame(episodes.loc[i], k) for k in image_keys], 1, size) for i in indices])


def load_lerobots(repo_ids, image_keys):
    """ load_lerobot for one dataset or a list of them (each with its own camera, image_keys: one key or a list),
    joined into one: frame, episode and task indices are offset
    (tasks with the same goal text share an index), and each episode records the dataset it came from. The datasets
    can have different frame rates (Bridge is 5 Hz, LIBERO and ManiSkill 20 Hz): each episode keeps its own. """
    import datasets, pandas as pd
    repo_ids = [repo_ids] if isinstance(repo_ids, str) else list(repo_ids)
    image_keys = [image_keys] * len(repo_ids) if isinstance(image_keys, str) else list(image_keys)
    all_frames, all_episodes, tasks = [], [], []
    n_frames = n_episodes = 0
    for repo_id, image_key in zip(repo_ids, image_keys, strict=True):
        frames, episodes, repo_tasks = load_lerobot(repo_id, image_key)
        task_map = []
        for t in repo_tasks:
            if t not in tasks:
                tasks.append(t)
            task_map.append(tasks.index(t))
        cols = frames.with_format("numpy")[:]
        frames = frames.remove_columns(["task_index", "episode_index"])
        frames = frames.add_column("task_index", np.asarray(task_map)[cols["task_index"]])
        frames = frames.add_column("episode_index", cols["episode_index"] + n_episodes)
        episodes = episodes.assign(episode_index=episodes["episode_index"] + n_episodes,
                                   dataset_from_index=episodes["dataset_from_index"] + n_frames, dataset=repo_id)
        n_frames, n_episodes = n_frames + len(frames), n_episodes + len(episodes)
        all_frames.append(frames); all_episodes.append(episodes)
        print(f"robot dataset {repo_id}: {len(episodes)} episodes, {len(frames)} frames, {len(repo_tasks)} tasks, {episodes['fps'][0]} Hz")
    ## Robots have different numbers of joints (the WidowX has 6, LIBERO's Panda 7): pad each dataset's joint positions
    ## with zeros to the longest, so the datasets can be joined
    ## (the lengths come from the column types: reading a whole column takes minutes for Bridge's 1.3M frames)
    longest = max(all_frames, key=lambda f: f.features[POSE_KEY].length)
    pose_dim = longest.features[POSE_KEY].length
    for i, f in enumerate(all_frames):
        if f.features[POSE_KEY].length < pose_dim:
            poses = np.stack(f.with_format("numpy")[:][POSE_KEY])
            poses = np.pad(poses, ((0, 0), (0, pose_dim - poses.shape[1])))
            f = f.remove_columns(POSE_KEY).add_column(POSE_KEY, list(poses))
            all_frames[i] = f.cast_column(POSE_KEY, longest.features[POSE_KEY]) ## the same fixed-length type, to join them
    frames = all_frames[0] if len(all_frames) == 1 else datasets.concatenate_datasets(all_frames)
    return frames, pd.concat(all_episodes, ignore_index=True), tasks


def episode_probabilities(episodes, weights):
    """ The probability of sampling each episode (a row of `episodes`): its dataset's share of the robot data mix
    (weights: {dataset: probability}, from dataset.robot_dataset_weights), split evenly over that dataset's episodes,
    so a dataset's share doesn't depend on how many episodes it has. Datasets with no episodes here are left out and the
    others' shares renormalized (e.g. a small val split). """
    counts = episodes["dataset"].value_counts()
    p = np.array([weights[d] / counts[d] for d in episodes["dataset"]])
    return p / p.sum()


class RobotEpisodes(torch.utils.data.Dataset):
    """ One item per episode: k random frames from it (distinct when the episode is long enough). Sampling
    trajectories first gives every trajectory the same weight, however long it is. Taking k frames per trajectory
    means one video open decodes k samples, at the cost of less diversity in a batch. """

    def __init__(self, frames, episodes, k, chunk):
        self._frames, self._episodes, self._k, self._chunk = frames, episodes, k, chunk

    def __len__(self):
        return len(self._episodes)

    def __getitem__(self, i):
        ep = self._episodes.iloc[i]
        n = int(ep["length"])
        ## torch RNG is seeded per worker
        offsets = torch.randperm(n)[:self._k] if n >= self._k else torch.randint(n, (self._k,))
        rows = self._frames[[int(ep["dataset_from_index"]) + int(j) for j in offsets]] ## dict of columns
        assert all(e == ep["episode_index"] for e in rows["episode_index"])
        rows["video"], rows["start"], rows["fps"] = ep["video"], ep["start"], ep["fps"]
        rows["dataset"] = ep["dataset"]
        ## the action chunk of each frame, flattened
        hold = hold_steps(ep["fps"]) ## each action held for this many steps of the chunk (1 at 20 Hz)
        chunks = self._frames[[int(ep["dataset_from_index"]) + min(int(j) + c // hold, n - 1) for j in offsets for c in range(self._chunk)]]["action"]
        rows["action"] = [sum(chunks[i:i + self._chunk], []) for i in range(0, len(chunks), self._chunk)]
        return rows


def side_by_side(per_camera, n, size):
    """ The frames of each camera, (n, H, W, 3) arrays or None for a camera the dataset doesn't have (blank: zeros),
    side by side: (n, H, W * cameras, 3). """
    return np.concatenate([x if x is not None else np.zeros((n, size[1], size[0], 3), np.uint8) for x in per_camera], axis=2)


def decode_frames(path, times, fps):
    """ Decodes the frames at the given times (seconds) of an MP4 file (one per episode) as RGB uint8 arrays,
    in the order of times. The file is opened once; each frame is a seek to the key frame at or before it
    (LeRobot puts one every 2 frames), which is cheaper than decoding through the frames in between. """
    import av
    out = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        ## One decoder thread: starting the default thread pool on every open costs more than decoding small frames
        stream.codec_context.thread_count = 1
        for t in times:
            container.seek(int(t / stream.time_base), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.time >= t - 0.5 / fps:
                    out.append(frame.to_ndarray(format="rgb24"))
                    break
    return out


class RobotCollate:
    """ Turns LeRobot frames into (X_text, X_img, Y, A, P): goal text, camera image and whitened pose as input, the
    whitened action as target. """

    def __init__(self, cfg, encode_txt, tasks):
        self._block = cfg.max_block_size
        self._size = (cfg.image_shape[0], cfg.image_shape[1])
        self._encode = encode_txt
        self._tasks = tasks
        self._chunk = cfg.policy.action_stacking
        ## {dataset: (action mean, action std)}, repeated over an action chunk
        self._action_stats = {repo_id: (torch.tensor(s["mean"]).repeat(cfg.policy.action_stacking), torch.tensor(s["std"]).repeat(cfg.policy.action_stacking))
                              for repo_id, s in cfg.dataset.action_stats.items()}
        self._pose_mean, self._pose_std = torch.tensor(cfg.dataset.pose_mean), torch.tensor(cfg.dataset.pose_std)

    def __call__(self, episodes):
        import cv2
        images = [img for ep in episodes for img in side_by_side(
            [np.stack([cv2.resize(img, self._size) for img in decode_frames(video, [start + t for t in ep["timestamp"]], ep["fps"])])
             if video is not None else None for video, start in zip(ep["video"], ep["start"])], len(ep["timestamp"]), self._size)]
        ## The first max_block_size characters of the goal, padded with spaces
        txt = [self._encode(self._tasks[t][:self._block].ljust(self._block)) for ep in episodes for t in ep["task_index"]]
        ## each frame's action chunk (k, chunk * action_dim), converted per action (to_batch_actions)
        chunks = [torch.tensor(ep["action"], dtype=torch.float32) for ep in episodes]
        chunks = [to_batch_actions(c.reshape(-1, c.shape[1] // self._chunk), ep["dataset"], ep["fps"]).reshape(c.shape)
                  for c, ep in zip(chunks, episodes)]
        actions = torch.cat([(c - self._action_stats[ep["dataset"]][0]) / self._action_stats[ep["dataset"]][1]
                             for c, ep in zip(chunks, episodes)])
        poses = torch.tensor([p for ep in episodes for p in ep[POSE_KEY]], dtype=torch.float32)
        return (torch.tensor(txt, dtype=torch.long), torch.from_numpy(np.stack(images)),
                torch.full((len(images),), -1, dtype=torch.long), ## -1 marks the row for the action head
                actions, (poses - self._pose_mean) / self._pose_std)


def decode_episode(path, start, length, fps, size):
    """ The `length` frames of one episode as a (length, H, W, 3) RGB uint8 array. The episode starts `start` seconds
    into its MP4 file, after the other episodes the file may hold. """
    import av
    out = np.zeros((length, size[1], size[0], 3), dtype=np.uint8)
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        container.seek(int(start / stream.time_base), stream=stream, backward=True)
        n = 0
        for frame in container.decode(stream):
            if frame.time < start - 0.5 / fps: ## from the key frame before the start: the end of the previous episode
                continue
            out[n] = to_rgb(frame.to_ndarray(format="rgb24"), size)
            n += 1
            if n == length:
                break
    assert n == length, f"{path}: {n} frames from {start} s, expected {length}"
    return out


class RobotCache:
    """ A cache of robot trajectories kept on `device` (the GPU, or the CPU): the images (frames, H, W, 3), actions and
    poses of all its slots in one buffer each, slot i's frames from start[i], plus each slot's length and goal. When
    every episode fits, they are packed one after the other; when episodes are swapped in, each slot has room for the
    longest episode. (Padding every episode to the longest instead would take 276 GB for LIBERO + PushT + Bridge,
    whose 2.1M frames take 26 GB.) sample() picks n_ep trajectories, each with its probability from
    episode_probabilities (its dataset's share of the mix, split over that dataset's slots), and k distinct frames in
    each, with tensor ops, so batches cost no CPU decoding.
    With fewer slots than episodes, random episodes keep being swapped into random slots while training samples from
    the cache. Training has priority: a DataLoader worker process at the lowest OS priority decodes them (in its own
    process, so decoding never holds the training thread's GIL), and a thread copies each to the device on a side
    stream, so the swap itself is one device-to-device copy that doesn't wait for queued training kernels. A lock
    orders it against sample(), so a batch never sees a half-written trajectory. """

    def __init__(self, cfg, encode_txt, frames, episodes, tasks, device, slots, weights, future_steps=0, future_actions=False, future_rewards=False):
        import random, threading
        self._episodes, self._size = episodes, (cfg.image_shape[0], cfg.image_shape[1])
        ## Actions, poses (whitened) and goals of every frame, read once, so a swap only has to decode the video. Actions
        ## are whitened per episode, with its dataset's stats
        cols = frames.select_columns(["action", POSE_KEY, "task_index", "episode_index"]).with_format("numpy")[:]
        self._actions = torch.tensor(np.stack(cols["action"]), dtype=torch.float32)
        self._action_stats = {repo_id: (torch.tensor(s["mean"]), torch.tensor(s["std"])) for repo_id, s in cfg.dataset.action_stats.items()}
        self._poses = (torch.tensor(np.stack(cols[POSE_KEY]), dtype=torch.float32) - torch.tensor(cfg.dataset.pose_mean)) / torch.tensor(cfg.dataset.pose_std)
        self._task, self._episode_index = cols["task_index"], cols["episode_index"]
        ## The first max_block_size characters of each goal, padded with spaces
        self._task_txt = torch.tensor([encode_txt(t[:cfg.max_block_size].ljust(cfg.max_block_size)) for t in tasks], device=device)
        n, self._max_length = min(slots, len(episodes)), int(episodes["length"].max())
        self._rng = random.Random(cfg.r_seed)
        ## Episode (row of `episodes`) held by each slot: all of them in order when they fit, otherwise a random subset
        self._in_cache = list(range(n)) if n == len(episodes) else self._rng.sample(range(len(episodes)), n)
        ## Probability of sampling each slot. A swap replaces an episode with one of the same dataset
        ## (_swap_forever), so each dataset keeps its slots and its share of the mix.
        cached = episodes.iloc[self._in_cache]
        assert set(cached["dataset"]) == set(episodes["dataset"]), "a dataset has no slot in the robot cache: raise dataset.cache_episodes"
        self.slot_p = torch.tensor(episode_probabilities(cached, weights), dtype=torch.float32, device=device)
        self._slots_of = {d: np.flatnonzero(cached["dataset"].to_numpy() == d).tolist() for d in set(cached["dataset"])}
        ## The same, restricted to one dataset's slots, to measure the loss on each dataset (sample(dataset=...))
        self.dataset_p = {d: self.slot_p * torch.isin(torch.arange(n, device=device), torch.tensor(slots, device=device))
                          for d, slots in self._slots_of.items()}
        ## Room per slot: its episode's length when they all fit (packed), otherwise the longest (any can be swapped in)
        room = episodes["length"].to_numpy() if n == len(episodes) else np.full(n, self._max_length)
        self._start = np.concatenate([[0], np.cumsum(room)[:-1]]).astype(np.int64) ## for _put, without a device sync
        self.start = torch.tensor(self._start, device=device)
        frames_in_cache = int(room.sum())
        cameras = len(episodes["video"].iloc[0]) ## shown side by side
        self.actions = torch.zeros((frames_in_cache, cfg.action_dim), device=device)
        self.poses = torch.zeros((frames_in_cache, cfg.pose_dim), device=device)
        self.goal_txt = torch.zeros((n, cfg.max_block_size), dtype=torch.long, device=device)
        self.length = torch.ones(n, dtype=torch.long, device=device)
        self.hold = torch.ones(n, dtype=torch.long, device=device) ## hold_steps of each slot's episode
        self.device, self.swaps, self._lock = device, 0, threading.Lock()
        self._chunk = torch.arange(cfg.policy.action_stacking, device=device) ## offsets of an action chunk
        ## sample() also returns the frames after the next future_steps actions (mini_wam.py: the chunk's, mini_dreamer.py:
        ## its sequence), with future_actions those actions too (mini_dreamer.py), and with future_rewards the reward after
        ## each (mini_tdmpc2.py)
        self._future = torch.arange(future_steps, device=device)
        self.future_actions, self.future_rewards = future_actions, future_rewards
        ## The frames are decoded by one worker process per CPU, each writing its episodes straight into a shared-memory
        ## buffer: in threads, PyAV holds the GIL too much to beat one thread, and passing each episode's frames back
        ## to this process costs as much as decoding them (log 2026-10-07). Then they go to the device (a copy only for cuda).
        ## empty, not zeros: share_memory_ copies the buffer, and a zeroed one would be in RAM twice for that copy (log
        ## 2026-10-08). Every frame row gets written: by the fill, or past a short episode never sampled.
        images = torch.empty((frames_in_cache, self._size[1], self._size[0] * cameras, 3), dtype=torch.uint8).share_memory_()
        fill_start = time.time()
        fill = DataLoader(_EpisodesIntoBuffer(episodes, self._in_cache, self._start, images, self._size), batch_size=None,
                          num_workers=len(os.sched_getaffinity(0)))
        for slot in fill:
            self._put(slot, *self._load(self._in_cache[slot]))
        self.decode_s = time.time() - fill_start ## logged as startup/<split>_cache_decode_s
        self.images = images.to(device)
        self._stop, self._thread = threading.Event(), None
        if n < len(episodes):
            ## Episodes in random order, forever. The workers are forked here, before torch.compile starts its threads.
            sampler = torch.utils.data.RandomSampler(range(len(episodes)), replacement=True, num_samples=2 ** 62)
            self._decoded = iter(DataLoader(_EpisodeVideos(episodes, self._size), batch_size=None, sampler=sampler,
                                            num_workers=cfg.dataset.cache_swap_workers, prefetch_factor=2,
                                            worker_init_fn=lambda _: os.nice(19)))
            self._thread = threading.Thread(target=self._swap_forever, daemon=True)
            self._thread.start()

    def _load(self, i):
        """ Episode i (a row of `episodes`): its whitened actions and poses, goal and hold steps. """
        ep = self._episodes.iloc[i]
        rows = slice(int(ep["dataset_from_index"]), int(ep["dataset_from_index"]) + int(ep["length"]))
        assert (self._episode_index[rows] == ep["episode_index"]).all() and len(set(self._task[rows])) == 1
        mean, std = self._action_stats[ep["dataset"]]
        return (to_batch_actions(self._actions[rows], ep["dataset"], ep["fps"]) - mean) / std, self._poses[rows], int(self._task[rows.start]), hold_steps(ep["fps"])

    def _put(self, slot, actions, poses, task, hold, images=None):
        """ Writes an episode into a slot: its images too, unless a fill worker already did. """
        with self._lock:
            rows = slice(self._start[slot], self._start[slot] + len(actions))
            if images is not None:
                self.images[rows] = images.to(self.device, non_blocking=True)
            self.actions[rows] = actions.to(self.device, non_blocking=True)
            self.poses[rows] = poses.to(self.device, non_blocking=True)
            self.goal_txt[slot] = self._task_txt[task]
            self.length[slot] = len(actions)
            self.hold[slot] = hold

    def _swap_forever(self):
        side = torch.cuda.Stream() if self.images.is_cuda else None
        for i, images in self._decoded: ## waits (without the GIL) for the worker's next decoded episode
            if self._stop.is_set():
                return
            if i in self._in_cache:
                continue
            actions, poses, task, hold = self._load(i)
            if side is not None: ## copy to the device off the training stream, then wait here, not in training
                with torch.cuda.stream(side):
                    images, actions, poses = (x.pin_memory().to(self.device, non_blocking=True) for x in (images, actions, poses))
                side.synchronize()
            slot = self._rng.choice(self._slots_of[self._episodes.iloc[i]["dataset"]]) ## one of the same dataset
            self._put(slot, actions, poses, task, hold, images)
            self._in_cache[slot] = i
            self.swaps += 1

    def close(self):
        """ Stops the swapper thread (a thread still running at exit aborts Python) and its decoding worker. """
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            del self._decoded

    def sample(self, n_ep, k, dataset=None):
        with self._lock:
            p = self.slot_p if dataset is None else self.dataset_p[dataset]
            e = torch.multinomial(p, n_ep, replacement=True) ## trajectories, by their share of the mix
            length = self.length[e, None]
            ## k distinct random frames per trajectory: the k smallest of random scores, frames past the end scored out
            scores = torch.rand(n_ep, self._max_length, device=self.device)
            scores[torch.arange(scores.shape[1], device=self.device) >= length] = 2.0
            t = scores.topk(k, dim=1, largest=False).indices % length ## (n_ep, k); % only matters when length < k
            ## (n_ep, k, chunk): each action held for hold steps (slower datasets), the last action past the end
            chunk = torch.minimum(t[..., None] + self._chunk // self.hold[e, None, None], length[..., None] - 1)
            ## (n_ep, k, future_steps): the next future_steps actions, as the chunk's
            future = torch.minimum(t[..., None] + self._future // self.hold[e, None, None], length[..., None] - 1)
            start = self.start[e, None] ## (n_ep, 1): where each trajectory's frames start in the buffers
            e = e[:, None].expand(-1, k)
            batch = (self.goal_txt[e].flatten(0, 1), self.images[start + t].flatten(0, 1),
                     torch.full((n_ep * k,), -1, dtype=torch.long, device=self.device),
                     self.actions[start[..., None] + chunk].flatten(2).flatten(0, 1), ## (n_ep * k, chunk * action_dim)
                     self.poses[start + t].flatten(0, 1))
            if not len(self._future):
                return batch
            ## The frame after each of those actions, the last frame past the end, stacked in the channels:
            ## (n_ep * k, H, W, future_steps * 3)
            next_frames = torch.minimum(future + 1, length[..., None] - 1)
            batch += (self.images[start[..., None] + next_frames].permute(0, 1, 3, 4, 2, 5).flatten(4).flatten(0, 1),)
            if not self.future_actions:
                return batch
            batch += (self.actions[start[..., None] + future].flatten(2).flatten(0, 1),) ## (n_ep * k, future_steps * action_dim)
            if not self.future_rewards:
                return batch
            ## 1 after each action from which the episode's last frame is reached: the demonstration has reached its goal
            return (*batch, (next_frames == length[..., None] - 1).float().flatten(0, 1)) ## (n_ep * k, future_steps)


class _EpisodeVideos(torch.utils.data.Dataset):
    """ Episode i (a row of `episodes`) decoded, its cameras side by side (runs in the worker process that swaps
    episodes into a RobotCache). """

    def __init__(self, episodes, size):
        self._episodes, self._size = episodes[["video", "start", "length", "fps"]], size

    def __len__(self):
        return len(self._episodes)

    def __getitem__(self, i):
        return i, torch.from_numpy(decode_cameras(self._episodes.iloc[i], self._size))


class _EpisodesIntoBuffer(torch.utils.data.Dataset):
    """ Item j decodes episode order[j] (a row of `episodes`) into rows start[j]... of `images`, a shared-memory buffer,
    and returns j (runs in the worker processes that fill a RobotCache). """

    def __init__(self, episodes, order, start, images, size):
        self._episodes, self._order, self._start = episodes[["video", "start", "length", "fps"]], order, start
        self._images, self._size = images, size

    def __len__(self):
        return len(self._order)

    def __getitem__(self, j):
        frames = decode_cameras(self._episodes.iloc[self._order[j]], self._size)
        self._images[self._start[j]: self._start[j] + len(frames)] = torch.from_numpy(frames)
        return j


def decode_cameras(ep, size):
    """ The frames of episode `ep` (a row of RobotCache's episodes), its cameras side by side. """
    length = int(ep["length"])
    return side_by_side([decode_episode(video, start, length, ep["fps"], size) if video is not None else None
                         for video, start in zip(ep["video"], ep["start"])], length, size)


def make_cache_loader(sample, *args):
    while True:
        yield sample(*args)


def make_loader(dataset, cfg, collate, num_workers, batch_size, sampler=None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, drop_last=True,
                        collate_fn=collate, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)
    while True: ## Loop over the data forever
        for batch in loader:
            yield batch


def n_cameras(cfg):
    """ The cameras side by side in each robot image: dataset.cameras, or one (dataset.robot_image_key). """
    return len(cfg.dataset.cameras) if cfg.dataset.get("cameras") else 1


def scale_image(img, bounds):
    """ uint8 pixels (0-255) to floats in bounds = [low, high], e.g. [0, 1] or [-1, 1]. """
    low, high = bounds
    return img / 255.0 * (high - low) + low


def to_device(batch, cfg):
    ## Pinned first: from pageable memory the copy can't be asynchronous, so it would wait for the GPU to finish the
    ## training step queued before it, instead of the CPU sampling the next batch meanwhile (log 2026-10-08)
    pin = lambda t: t.pin_memory() if t.device.type == "cpu" and str(cfg.device).startswith("cuda") else t
    ## future: a RobotCache's future frames, and their actions with future_actions
    X_text, X_img, Y, A, P, *future = (pin(t).to(cfg.device, non_blocking=True) for t in batch)
    return X_text, scale_image(X_img.float(), cfg.image_bounds), Y, A, P, *(scale_image(f.float(), cfg.image_bounds) for f in future[:1]), *future[1:]

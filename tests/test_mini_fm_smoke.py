import glob
import importlib.util
import math
import os
import re
import subprocess
import sys
import time
import unittest
from pathlib import Path

"""Smoke tests for mini_fm.py's models, trained by mini_train.py, one per model.type and action head.

Each test trains for a few hundred steps on libero_spatial robot data,
then runs one short in-process LIBERO sim eval, and checks that:
1) training finishes and logs finite action-head losses (mse, l1, huber or diffusion),
2) the sim eval reports a success rate,
3) a rollout MP4 is saved in the run folder,
4) the best model by sim success, miniFM_best.pth, is saved there.

The convnet tests cover ConvNetFM (mini-GRP's convnet behind mini_fm.py's interface, conditioned on the goal text).
The ManiSkill and Bridge ManiSkill tests are skipped without ManiSkill or the Bridge assets. These are pipeline
checks, not success checks: a few hundred steps are too few for the policy to solve a task.

Every run gets its own folder, outputs/tests/<test>_<time>/, which is kept.
"""

REPO = Path(__file__).resolve().parents[1]  # mini-wm
## The ManiSkill tests need ManiSkill, and the Bridge ManiSkill ones its Bridge assets
## (python -m mani_skill.utils.download_asset bridge_v2_real2sim); without them they are skipped
MANISKILL = importlib.util.find_spec("mani_skill") is not None
BRIDGE_ASSETS = (Path(os.environ.get("MS_ASSET_DIR", Path.home() / ".maniskill")) / "data" / "tasks" / "bridge_v2_real2sim_dataset").exists()
needs_maniskill = unittest.skipUnless(MANISKILL, "ManiSkill isn't installed")
needs_bridge = unittest.skipUnless(MANISKILL and BRIDGE_ASSETS, "Bridge ManiSkill's assets aren't downloaded")
ONE_CAMERA = ("dataset.cameras=null", "image_block_size=64")  # the convnet takes one camera, not mini-FM's default two
## The Bridge twins the training evals run (conf/sim/full.yaml), not all of the bridge task set's: testing.yaml
## lists none, which would also run PutEggplantInBasket
BRIDGE_TWINS = ("sim.simulators.maniskill.task_sets={bridge:{tasks:[PutCarrotOnPlateInScene-v1,PutSpoonOnTableClothInScene-v1,"
                "StackGreenCubeOnYellowCubeBakedTexInScene-v1]}}")


def run_mini_fm(name, *overrides, config="tests/mini-fm-testing"):
    """Runs mini_train.py with a short test config; returns (returncode, output, run_dir)."""
    run_dir = REPO / "outputs" / "tests" / f"{name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    cmd = [
        sys.executable, "mini_train.py",
        "--config-name", config,  # e.g. conf/tests/mini-fm-testing.yaml
        f"hydra.run.dir={run_dir}",
        *overrides,
    ]
    env = {**os.environ, "MUJOCO_GL": "egl"}
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=3600, env=env)
    return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or ""), run_dir


def descendants(pid):
    """The process ids under pid (Linux /proc)."""
    children = [int(c) for task in glob.glob(f"/proc/{pid}/task/*/children") for c in Path(task).read_text().split()]
    return children + [d for c in children for d in descendants(c)]


def parent(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])


def cmdline(pid):
    return Path(f"/proc/{pid}/cmdline").read_text().replace("\0", " ")


class TestMiniFMSmoke(unittest.TestCase):
    def _check_run(self, name, *overrides):
        code, output, run_dir = run_mini_fm(name, *overrides)
        self.assertEqual(code, 0, msg=f"mini_train.py failed with code {code}. Output:\n{output[-5000:]}")

        action_loss = [float(m) for m in re.findall(r"val (?:mse|l1|huber|diffusion) ([-\d.naif]+)", output)]
        self.assertTrue(action_loss, msg=f"No val action loss logged. Output:\n{output[-5000:]}")
        self.assertTrue(all(math.isfinite(v) for v in action_loss), msg=f"Non-finite val action loss: {action_loss}")

        self.assertRegex(output, r"sim success rate libero_spatial \d", msg=f"No sim eval. Output:\n{output[-5000:]}")
        videos = glob.glob(str(run_dir / "sim_eval" / "libero_spatial" / "*.mp4"))
        self.assertTrue(videos, msg=f"No rollout video in {run_dir}/sim_eval/libero_spatial")
        self.assertTrue((run_dir / "miniFM_best.pth").exists(), msg=f"No best model saved in {run_dir}")
        return output, run_dir

    def test_mini_fm_fm_robot_smoke(self):
        """The transformer (FM) on robot data, with a sim eval."""
        self._check_run("fm_robotfrac1.0_chunk5_wrist", "model.type=fm")

    def test_mini_fm_fm_cache_swap_smoke(self):
        """The robot cache with fewer slots than episodes (dataset.cache_episodes), so a worker keeps swapping episodes in."""
        output, _ = self._check_run("fm_robotfrac1.0_chunk5_wrist_cacheslots100", "model.type=fm", "dataset.cache_episodes=100")
        self.assertIn("robot train cache: 100 of", output, msg=output[-5000:])
        self.assertRegex(output, r"cache swaps [1-9]", msg=f"No episodes swapped in. Output:\n{output[-5000:]}")

    def test_mini_fm_fm_background_sim_eval_smoke(self):
        """The transformer with the sim eval in the background process (sim/sim_eval.py's BackgroundSimEval), as in real runs."""
        self._check_run("fm_robotfrac1.0_chunk5_wrist_backgroundsim", "model.type=fm", "sim.background=true")

    def test_mini_fm_fm_background_sim_eval_restart_smoke(self):
        """Kills a LIBERO env process during the first background sim eval (as MuJoCo's EGL renderer crashes now and
        then): training must restart the evaluator and still finish with a sim eval, not hang or skip the later ones."""
        run_dir = REPO / "outputs" / "tests" / f"fm_backgroundsim_restart_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        run_dir.mkdir(parents=True)
        cmd = [sys.executable, "mini_train.py", "--config-name", "tests/mini-fm-testing", f"hydra.run.dir={run_dir}",
               "model.type=fm", "sim.background=true", "sim.eval_interval=100"]  # evals at steps 100, 200 and 299
        with open(run_dir / "test_output.txt", "w") as log:
            proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "MUJOCO_GL": "egl"})
            killed = None
            while proc.poll() is None and killed is None:  # kill the first LIBERO env worker that shows up
                time.sleep(0.5)
                try:
                    workers = [p for p in descendants(proc.pid) if "forkserver" in cmdline(parent(p))]
                except OSError:  # a process exited while we looked
                    continue
                if workers:
                    killed = workers[0]
                    os.kill(killed, 9)
            proc.wait(timeout=1800)
        output = (run_dir / "test_output.txt").read_text()
        self.assertIsNotNone(killed, msg=f"No LIBERO env process found to kill. Output:\n{output[-5000:]}")
        self.assertEqual(proc.returncode, 0, msg=f"mini_train.py failed. Output:\n{output[-5000:]}")
        self.assertIn("restarting it", output, msg=f"The evaluator wasn't restarted. Output:\n{output[-5000:]}")
        self.assertRegex(output, r"sim success rate libero_spatial \d", msg=f"No sim eval after the restart. Output:\n{output[-5000:]}")

    def test_mini_fm_fm_diffusion_head_smoke(self):
        """The transformer with the diffusion action head, which samples each action chunk in the sim."""
        self._check_run("fm_robotfrac1.0_chunk5_wrist_diffusionhead", "model.type=fm", "model.action_head=diffusion")

    def test_mini_fm_fm_token_per_action_smoke(self):
        """The transformer with one class token per action of the chunk (model.token_per_action), each decoded by the
        same linear layer."""
        self._check_run("fm_robotfrac1.0_chunk5_wrist_tokenperaction", "model.type=fm", "model.token_per_action=true")

    def test_mini_fm_fm_l1_head_smoke(self):
        """The transformer with the L1 action head (model.action_head=l1)."""
        output, _ = self._check_run("fm_robotfrac1.0_chunk5_wrist_l1head", "model.type=fm", "model.action_head=l1")
        self.assertRegex(output, r"val l1 \d", msg=f"No val l1 loss. Output:\n{output[-5000:]}")

    def test_mini_fm_fm_token_per_action_diffusion_smoke(self):
        """The transformer with one causal token per action (model.token_per_action) and the diffusion head, conditioned
        on all the action tokens."""
        self._check_run("fm_robotfrac1.0_chunk5_wrist_tokenperaction_diffusionhead", "model.type=fm",
                        "model.token_per_action=true", "model.action_head=diffusion")

    def test_mini_fm_fm_mse_head_smoke(self):
        """The transformer with the MSE action head (model.action_head=mse; the default is huber)."""
        output, _ = self._check_run("fm_robotfrac1.0_chunk5_wrist_msehead", "model.type=fm", "model.action_head=mse")
        self.assertRegex(output, r"val mse \d", msg=f"No val mse loss. Output:\n{output[-5000:]}")

    def test_mini_fm_fm_color_jitter_smoke(self):
        """The transformer with colour jitter on its training images (augment_color), for the colour gap to the sims."""
        self._check_run("fm_robotfrac1.0_chunk5_wrist_augcolor1", "model.type=fm", "augment_color=1.0")

    def test_mini_fm_fm_one_camera_smoke(self):
        """The transformer on one camera per dataset (dataset.cameras=null, robot_image_key's), not the default two."""
        self._check_run("fm_robotfrac1.0_chunk5_wrist_onecamera", "model.type=fm", *ONE_CAMERA)

    @needs_bridge
    def test_mini_fm_fm_two_cameras_smoke(self):
        """The transformer seeing the wrist and agent-view cameras side by side (dataset.cameras), on LIBERO spatial
        (both cameras) and the PutSpoon twin's demos (agent view only, so a blank wrist image), with sim evals in LIBERO
        spatial and the Bridge twins."""
        output, run_dir = self._check_run("fm_robotfrac1.0_chunk5_twocameras_spatial_putspoon", "model.type=fm",
                                          "dataset.cameras=[observation.images.wrist_image,observation.images.image]", "image_block_size=128",
                                          "sim.simulators.maniskill.episodes=2",
                                          "dataset.robot_dataset_name=[gberseth/libero_spatial_no_noops_lerobot,gberseth/maniskill_putspoonontablecloth_lerobot]",
                                          "sim.task_sets=[libero_spatial,bridge]",
                                          "dataset.robot_image_key=[observation.images.wrist_image,observation.images.image]")
        self.assertIn("no camera ['observation.images.wrist_image'], left blank", output, msg=output[-5000:])
        self.assertRegex(output, r"bridge \d", msg=f"No Bridge twin sim eval. Output:\n{output[-5000:]}")

    def test_mini_fm_convnet_robot_smoke(self):
        """GRPConvNet, conditioned on the goal text without a goal image, with a sim eval."""
        self._check_run("convnet_robotfrac1.0_chunk5_wrist", "model.type=convnet", *ONE_CAMERA)

    def test_mini_fm_convnet_diffusion_head_smoke(self):
        """GRPConvNet with the diffusion action head, conditioned on its fused features instead of its MLP head."""
        self._check_run("convnet_robotfrac1.0_chunk5_wrist_diffusionhead", "model.type=convnet", *ONE_CAMERA, "model.action_head=diffusion")

    def test_mini_fm_fm_pose_smoke(self):
        """The transformer given the robot's pose (policy.use_pose_data), from the dataset in training and the sim in evals."""
        self._check_run("fm_robotfrac1.0_chunk5_wrist_pose", "model.type=fm", "policy.use_pose_data=true")

    def test_mini_fm_convnet_pose_smoke(self):
        """GRPConvNet given the robot's pose (policy.use_pose_data), through its pose embedding."""
        self._check_run("convnet_robotfrac1.0_chunk5_wrist_pose", "model.type=convnet", *ONE_CAMERA, "policy.use_pose_data=true")

    @needs_maniskill
    def test_mini_fm_fm_libero_maniskill_smoke(self):
        """The transformer on LIBERO spatial and ManiSkill PushT data together, each with its own action whitening stats
        and sampled equally (dataset.robot_dataset_weights), with sim evals in both."""
        output, run_dir = self._check_run("fm_robotfrac1.0_chunk5_wrist_spatial_pusht", "model.type=fm", "sim.simulators.maniskill.episodes=4",
                                          "dataset.robot_dataset_name=[gberseth/libero_spatial_no_noops_lerobot,gberseth/maniskill_pusht_lerobot]",
                                          "sim.task_sets=[libero_spatial,PushT-v1]")
        self.assertRegex(output, r"PushT-v1 \d", msg=f"No ManiSkill sim eval. Output:\n{output[-5000:]}")
        self.assertEqual(len(re.findall(r"action mean gberseth/", output)), 2, msg="Not one set of action stats per dataset")
        self.assertIn("robot data mix: libero_spatial_no_noops_lerobot 0.500, maniskill_pusht_lerobot 0.500", output,
                      msg="The two datasets are not sampled equally by default")
        self.assertTrue(glob.glob(str(run_dir / "sim_eval" / "PushT-v1" / "*.mp4")), msg=f"No PushT video in {run_dir}/sim_eval")

    @needs_maniskill
    def test_mini_fm_fm_maniskill_tabletop_smoke(self):
        """The transformer on LIBERO spatial and the ManiSkill Panda-stick tasks (PushT, DrawTriangle), with sim evals in
        both ManiSkill tabletop task sets: maniskill_panda_stick (no gripper) and maniskill_panda (11 tasks, the gripper's
        sign flipped back to ManiSkill's); the small stick dataset stands in for gberseth/maniskill_panda_lerobot."""
        output, run_dir = self._check_run("fm_robotfrac1.0_chunk5_wrist_spatial_panda_stick", "model.type=fm", "sim.simulators.maniskill.episodes=2",
                                          "dataset.robot_dataset_name=[gberseth/libero_spatial_no_noops_lerobot,gberseth/maniskill_panda_stick_lerobot]",
                                          "sim.task_sets=[libero_spatial,maniskill_panda_stick,maniskill_panda]")
        for task_set, n in [("maniskill_panda_stick", 2), ("maniskill_panda", 11)]:
            self.assertRegex(output, rf"{task_set} \d", msg=f"No {task_set} sim eval. Output:\n{output[-5000:]}")
            self.assertEqual(len(glob.glob(str(run_dir / "sim_eval" / task_set / "*-task-id-*.mp4"))) % n, 0,
                             msg=f"Not one video per task in {run_dir}/sim_eval/{task_set}")

    def test_mini_fm_fm_pushletter_smoke(self):
        """The transformer on LIBERO spatial (wrist camera, Panda's 7 joints) and ManiSkill's WidowX PushLetter (agent
        view, 6 joints, zero-padded to 7) together, given the pose. PushLetter's env isn't in ManiSkill, so only LIBERO
        spatial is evaluated."""
        output, _ = self._check_run("fm_robotfrac1.0_chunk5_spatialwrist_pushletteragentview_pose", "model.type=fm",
                                    "policy.use_pose_data=true",
                                    "dataset.robot_dataset_name=[gberseth/libero_spatial_no_noops_lerobot,gberseth/maniskill_pushletter_lerobot]",
                                    "dataset.robot_image_key=[observation.images.wrist_image,observation.images.image]")
        self.assertIn("robot data mix: libero_spatial_no_noops_lerobot 0.500, maniskill_pushletter_lerobot 0.500", output,
                      msg="The two datasets are not sampled equally by default")
        sim_lines = re.findall(r"sim success rate .*", output)
        self.assertTrue(sim_lines and all("pushletter" not in line for line in sim_lines),
                        msg=f"PushLetter has no sim, but was evaluated: {sim_lines}")

    @needs_bridge
    def test_mini_fm_fm_widowx_twin_demos_smoke(self):
        """The transformer on LIBERO spatial (wrist camera) and the demos recorded in ManiSkill's PutSpoon, PutCarrot and
        StackCube Bridge twins (agent view, 5 Hz, converted to Bridge's labels: data_loading.BRIDGE_TWIN_DEMOS), with sim
        evals in LIBERO spatial and the Bridge twins (the bridge task set, which these datasets map to)."""
        twins = ["maniskill_putspoonontablecloth_lerobot", "maniskill_putcarrotonplate_lerobot", "maniskill_stackcube_lerobot"]
        output, run_dir = self._check_run("fm_robotfrac1.0_chunk5_spatialwrist_widowxtwindemosagentview", "model.type=fm",
                                          "sim.simulators.maniskill.episodes=2",
                                          f"dataset.robot_dataset_name=[gberseth/libero_spatial_no_noops_lerobot,{','.join('gberseth/' + d for d in twins)}]",
                                          "sim.task_sets=[libero_spatial,bridge]", BRIDGE_TWINS,
                                          "dataset.robot_image_key=[observation.images.wrist_image" + ",observation.images.image" * 3 + "]")
        self.assertRegex(output, r"bridge \d", msg=f"No Bridge twin sim eval. Output:\n{output[-5000:]}")
        self.assertEqual(len(glob.glob(str(run_dir / "sim_eval" / "bridge" / "*-task-id-*.mp4"))), 3,
                         msg=f"Not one video per Bridge twin in {run_dir}/sim_eval/bridge")

    @needs_bridge
    def test_mini_fm_fm_bridge_twins_smoke(self):
        """The transformer on the agent view, with sim evals in LIBERO spatial and three of ManiSkill's Bridge digital twins
        (the WidowX scenes that policies trained on gberseth/bridge_orig_lerobot are evaluated in). Trains on LIBERO
        spatial only, to keep the download small: this checks the eval runs, not that it succeeds."""
        output, run_dir = self._check_run("fm_robotfrac1.0_chunk5_agentview_spatial_bridgetwins", "model.type=fm",
                                          "dataset.robot_image_key=observation.images.image", "sim.simulators.maniskill.episodes=2",
                                          "sim.task_sets=[libero_spatial,bridge]", BRIDGE_TWINS)
        self.assertRegex(output, r"bridge \d", msg=f"No Bridge twin sim eval. Output:\n{output[-5000:]}")
        self.assertEqual(len(glob.glob(str(run_dir / "sim_eval" / "bridge" / "*-task-id-*.mp4"))) % 3, 0,
                         msg=f"Not one video per Bridge twin in {run_dir}/sim_eval/bridge")

    def test_mini_fm_fm_image_bounds_minus1_1_smoke(self):
        """The transformer with pixels scaled to [-1, 1] (image_bounds) instead of [0, 1], in training and the sim."""
        self._check_run("fm_robotfrac1.0_chunk5_wrist_imgbounds-1to1", "model.type=fm", "image_bounds=[-1.0,1.0]")


if __name__ == "__main__":
    unittest.main()

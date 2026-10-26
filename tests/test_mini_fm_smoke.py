import glob
import math
import os
import re
import subprocess
import sys
import time
import unittest
from pathlib import Path

"""Smoke tests for mini_fm.py's models, trained by mini_train.py, one per model.type and action head.

Each test trains for a few hundred steps on libero_spatial robot data (the short run of
conf/tests/mini-fm-testing.yaml), then runs one short in-process LIBERO sim eval, and checks that:
1) training finishes and logs finite action-head losses (mse, l1, huber or diffusion),
2) the sim eval reports a success rate,
3) a rollout MP4 is saved in the run folder,
4) the best model by sim success, miniFM_best.pth, is saved there.
These are pipeline checks, not success checks: a few hundred steps are too few for the policy to solve a task.

Every run gets its own folder, outputs/tests/<test>_<time>/, which is kept.
"""

REPO = Path(__file__).resolve().parents[1]
ONE_CAMERA = ("dataset.cameras=null", "image_block_size=64")  # one camera instead of mini-FM's default two


def run_mini_fm(name, *overrides, config="tests/mini-fm-testing"):
    """Runs mini_train.py with a short test config; returns (returncode, output, run_dir)."""
    run_dir = REPO / "outputs" / "tests" / f"{name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    cmd = [sys.executable, "mini_train.py", "--config-name", config, f"hydra.run.dir={run_dir}", *overrides]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=3600, env={**os.environ, "MUJOCO_GL": "egl"})
    return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or ""), run_dir


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
        """The transformer (FM), with the Huber action head and two cameras: the quick check after a code change."""
        self._check_run("fm_chunk5_wristagentview", "model.type=fm")

    def test_mini_fm_fm_diffusion_head_smoke(self):
        """The transformer with the diffusion action head, which samples each action chunk in the sim."""
        self._check_run("fm_chunk5_wristagentview_diffusionhead", "model.type=fm", "model.action_head=diffusion")

    def test_mini_fm_fm_token_per_action_smoke(self):
        """One class token per action of the chunk (model.token_per_action), each decoded by the same linear layer."""
        self._check_run("fm_chunk5_wristagentview_tokenperaction", "model.type=fm", "model.token_per_action=true")

    def test_mini_fm_fm_pose_smoke(self):
        """The transformer given the robot's pose, its arm joint positions."""
        self._check_run("fm_chunk5_wristagentview_pose", "model.type=fm", "policy.use_pose_data=true")

    def test_mini_fm_fm_one_camera_smoke(self):
        """The transformer on the wrist camera alone."""
        self._check_run("fm_chunk5_wrist", "model.type=fm", *ONE_CAMERA)

    def test_mini_fm_small_convnet_smoke(self):
        """The three-convolution policy (SmallConvNetFM), on one camera."""
        self._check_run("small_convnet_chunk5_wrist", "model.type=small_convnet", *ONE_CAMERA)

    def test_mini_fm_fm_background_sim_eval_smoke(self):
        """The transformer with the sim eval in the background process (sim_eval.py's BackgroundSimEval), as in real runs."""
        self._check_run("fm_chunk5_wristagentview_backgroundsim", "model.type=fm", "sim.background=true")


if __name__ == "__main__":
    unittest.main()

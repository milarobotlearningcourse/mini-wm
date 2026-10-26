import glob
import math
import re
import unittest

from test_mini_fm_smoke import run_mini_fm

"""Smoke test for mini_wam.py, the world action model, trained by mini_train.py: the short run of
conf/tests/mini-wam-testing.yaml (300 steps on libero_spatial, then one in-process sim eval) must
1) log finite val losses of the action head (mse) and of the frame diffusion (frames), overall and by dataset,
2) save a grid of predicted frames (predicted_frames/<step>.png, every model.frame_image_interval steps and the last),
3) run the sim eval and save its rollout video and the best model, miniWAM_best.pth.
A pipeline check, not a quality check: 300 steps are too few to predict sharp frames or solve a task.
Every run gets its own folder, outputs/tests/<test>_<time>/, which is kept.
"""


class TestMiniWAMSmoke(unittest.TestCase):
    def _check_run(self, name, *overrides):
        code, output, run_dir = run_mini_fm(name, *overrides, config="tests/mini-wam-testing")
        self.assertEqual(code, 0, msg=f"mini_train.py failed with code {code}. Output:\n{output[-5000:]}")
        for head in ("mse", "frames"):
            losses = [float(m) for m in re.findall(rf"val {head} ([-\d.naif]+)", output)]
            self.assertTrue(losses and all(math.isfinite(v) for v in losses), msg=f"val {head} losses: {losses}\n{output[-5000:]}")
        self.assertIn("val frames loss by dataset: libero_spatial_no_noops_lerobot", output, msg=output[-5000:])
        self.assertTrue(glob.glob(str(run_dir / "predicted_frames" / "*.png")), msg=f"No predicted frames in {run_dir}")
        self.assertRegex(output, r"sim success rate libero_spatial \d", msg=f"No sim eval. Output:\n{output[-5000:]}")
        self.assertTrue(glob.glob(str(run_dir / "sim_eval" / "libero_spatial" / "*.mp4")), msg=f"No rollout video in {run_dir}")
        self.assertTrue((run_dir / "miniWAM_best.pth").exists(), msg=f"No best model saved in {run_dir}")

    def test_mini_wam_robot_smoke(self):
        self._check_run("wam_robotfrac1.0_chunk5_wrist_frames32")

    def test_mini_wam_framepatch4_adaln_smoke(self):
        """Smaller frame patches (model.frame_patch_size 4: 64 tokens per frame) and adaLN-Zero conditioning of the frame
        tokens on the noise step in every block (model.frame_adaln), as in DiT."""
        self._check_run("wam_robotfrac1.0_chunk5_wrist_frames32_framepatch4_adaln", "model.frame_patch_size=4", "model.frame_adaln=true")


if __name__ == "__main__":
    unittest.main()

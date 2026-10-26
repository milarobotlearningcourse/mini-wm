import glob
import math
import re
import unittest

from tests.test_mini_fm_smoke import run_mini_fm

"""Smoke test for mini_dreamer.py, the forward dynamics model p(s'|s, a) built as DreamerV3's world model
(the frames after the next model.sequence_steps actions, from the current frame), trained next to mini-FM's policy by
mini_train.py: the short run of conf/tests/mini-dreamer-testing.yaml (20 steps on libero_spatial, under 5 min, then one
in-process sim eval) must
1) log finite val losses of the action head (huber), of the decoded frames (frames), of the KL losses (kl_dyn, kl_rep), of the reward
   head (reward), of the critic (critic) and of mini-FM in imagination (actor), and the action and frame losses by
   dataset,
2) save a grid of predicted frames (predicted_frames/<step>.png, every model.frame_image_interval steps and the last),
3) run the sim eval and save its rollout video and the best model, miniDreamer_best.pth.
A pipeline check, not a quality check: 20 steps are too few to predict sharp frames or solve a task.
Every run gets its own folder, outputs/tests/<test>_<time>/, which is kept.
"""


class TestMiniDreamerSmoke(unittest.TestCase):
    def _check_run(self, name, *overrides):
        code, output, run_dir = run_mini_fm(name, *overrides, config="tests/mini-dreamer-testing")
        self.assertEqual(code, 0, msg=f"mini_train.py failed with code {code}. Output:\n{output[-5000:]}")
        for head in ("huber", "frames", "kl_dyn", "kl_rep", "reward", "critic", "actor"):
            losses = [float(m) for m in re.findall(rf"val {head} ([-\d.naif]+)", output)]
            self.assertTrue(losses and all(math.isfinite(v) for v in losses), msg=f"val {head} losses: {losses}\n{output[-5000:]}")
        self.assertIn("val frames loss by dataset: libero_spatial_no_noops_lerobot", output, msg=output[-5000:])
        self.assertTrue(glob.glob(str(run_dir / "predicted_frames" / "*.png")), msg=f"No predicted frames in {run_dir}")
        self.assertRegex(output, r"sim success rate libero_spatial \d", msg=f"No sim eval. Output:\n{output[-5000:]}")
        self.assertTrue(glob.glob(str(run_dir / "sim_eval" / "libero_spatial" / "*.mp4")), msg=f"No rollout video in {run_dir}")
        self.assertTrue((run_dir / "miniDreamer_best.pth").exists(), msg=f"No best model saved in {run_dir}")

    def test_mini_dreamer_robot_smoke(self):
        """DreamerV3's categorical latents (model.latent categorical, the default)."""
        self._check_run("dreamer_robotfrac1.0_chunk1_wrist_steps16_frames32_categorical32x32")

    def test_mini_dreamer_gaussian_smoke(self):
        """DreamerV1's Gaussian latents (model.latent gaussian)."""
        self._check_run("dreamer_robotfrac1.0_chunk1_wrist_steps16_frames32_gaussian32", "model.latent=gaussian")

    def test_mini_dreamer_small_convnet_smoke(self):
        """The three-convolution policy (model.policy_net small_convnet) in place of mini-FM's transformer."""
        self._check_run("dreamer_policysmall_convnet_robotfrac1.0_chunk1_wrist_steps16_frames32_categorical32x32",
                        "model.policy_net=small_convnet")


if __name__ == "__main__":
    unittest.main()

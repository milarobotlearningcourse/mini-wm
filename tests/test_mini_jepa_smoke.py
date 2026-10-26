import glob
import math
import re
import unittest

from test_mini_fm_smoke import run_mini_fm

"""Smoke tests for mini_jepa.py, the latent world model after RoboJEPA (the embeddings of the frames after the
next model.sequence_steps actions, from the current frame), trained by mini_train.py, acting with each of its
policies: planning with CEM towards a goal image (model.policy=cem) or mini-FM, trained next to it (model.policy=fm).
The short run of conf/tests/mini-jepa-testing.yaml (10 steps on libero_spatial, then one in-process sim eval; under 5 min) must
1) log finite val losses of the embeddings (latent_tf, latent_ar), and of the action head (huber) with fm, and the
   targets' spread (target_std) above 0, so the embeddings haven't collapsed,
2) save a grid of imagined frames (predicted_frames/<step>.png, every model.frame_image_interval steps and the last),
3) run the sim eval and save its rollout video and the best model, miniJEPA_best.pth.
A pipeline check, not a quality check: 300 steps are too few to predict the future well or solve a task.
Every run gets its own folder, outputs/tests/<test>_<time>/, which is kept.
"""


class TestMiniJEPASmoke(unittest.TestCase):
    def check_run(self, name, policy, heads):
        code, output, run_dir = run_mini_fm(name, f"model.policy={policy}", config="tests/mini-jepa-testing")
        self.assertEqual(code, 0, msg=f"mini_train.py failed with code {code}. Output:\n{output[-5000:]}")
        for head in heads:
            losses = [float(m) for m in re.findall(rf"val {head} ([-\d.naif]+)", output)]
            self.assertTrue(losses and all(math.isfinite(v) for v in losses), msg=f"val {head} losses: {losses}\n{output[-5000:]}")
        self.assertGreater(float(re.findall(r"val target_std ([\d.]+)", output)[-1]), 0.01, msg="the embeddings collapsed")
        self.assertTrue(glob.glob(str(run_dir / "predicted_frames" / "*.png")), msg=f"No predicted frames in {run_dir}")
        self.assertRegex(output, r"sim success rate libero_spatial \d", msg=f"No sim eval. Output:\n{output[-5000:]}")
        self.assertTrue(glob.glob(str(run_dir / "sim_eval" / "libero_spatial" / "*.mp4")), msg=f"No rollout video in {run_dir}")
        self.assertTrue((run_dir / "miniJEPA_best.pth").exists(), msg=f"No best model saved in {run_dir}")

    def test_mini_jepa_robot_smoke(self):
        self.check_run("jepa_policycem_robotfrac1.0_chunk5_wrist_steps16_skip4_frames64", "cem", ("latent_tf", "latent_ar", "target_std"))

    def test_mini_jepa_fm_policy_smoke(self):
        self.check_run("jepa_policyfm_robotfrac1.0_chunk5_wrist_steps16_skip4_frames64", "fm", ("huber", "latent_tf", "latent_ar", "target_std"))


if __name__ == "__main__":
    unittest.main()

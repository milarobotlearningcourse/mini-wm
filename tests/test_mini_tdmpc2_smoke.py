import glob
import math
import re
import unittest

from tests.test_mini_fm_smoke import run_mini_fm

"""Smoke test for mini_tdmpc2.py, TD-MPC2 learned offline from the demonstrations (its own policy, which plans
with MPPI), trained by mini_train.py: the short run of conf/tests/mini-tdmpc2-testing.yaml (300 steps on
libero_spatial, then one in-process sim eval) must
1) log finite val losses of the latent consistency, the rewards, the Q-values and the policy, and some rewards (the
   demonstrations' ends) in the batches,
2) run the sim eval, with MPPI planning, and save its rollout video and the best model, miniTDMPC2_best.pth.
A pipeline check, not a quality check: 300 steps are too few to solve a task.
Every run gets its own folder, outputs/tests/<test>_<time>/, which is kept.
"""


class TestMiniTDMPC2Smoke(unittest.TestCase):
    def test_mini_tdmpc2_robot_smoke(self):
        code, output, run_dir = run_mini_fm("tdmpc2_robotfrac1.0_chunk5_wrist_horizon3", config="tests/mini-tdmpc2-testing")
        self.assertEqual(code, 0, msg=f"mini_train.py failed with code {code}. Output:\n{output[-5000:]}")
        for head in ("consistency", "reward_ce", "value_ce", "pi_loss", "q"):
            losses = [float(m) for m in re.findall(rf"val {head} ([-\d.naif]+)", output)]
            self.assertTrue(losses and all(math.isfinite(v) for v in losses), msg=f"val {head} losses: {losses}\n{output[-5000:]}")
        self.assertGreater(float(re.findall(r"val reward_rate ([\d.]+)", output)[-1]), 0, msg="no rewards in the batches")
        self.assertRegex(output, r"sim success rate libero_spatial \d", msg=f"No sim eval. Output:\n{output[-5000:]}")
        self.assertTrue(glob.glob(str(run_dir / "sim_eval" / "libero_spatial" / "*.mp4")), msg=f"No rollout video in {run_dir}")
        self.assertTrue((run_dir / "miniTDMPC2_best.pth").exists(), msg=f"No best model saved in {run_dir}")


if __name__ == "__main__":
    unittest.main()

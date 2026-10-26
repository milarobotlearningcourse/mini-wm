import glob
import re
import unittest

from test_mini_fm_smoke import run_mini_fm

"""Smoke test for mini_vlaps.py, Monte Carlo tree search with a pre-trained mini-Dreamer and its mini-FM, as
VLAPS: the short run of conf/tests/mini-dreamer-testing.yaml gives a miniDreamer_best.pth, then conf/mini-vlaps.yaml
searches with it, with a small search (2 simulations of 3 children), for one short episode (sim=testing) of task 0 of
libero_spatial, towards the last frame of the demo that started from init state 0. It must log the episode's result
and the success rate, and save the episode's video. A pipeline check, not a quality check.
Every run gets its own folder, outputs/tests/<test>_<time>/, which is kept.
"""


class TestMiniVLAPSSmoke(unittest.TestCase):
    def test_mini_vlaps_smoke(self):
        code, output, dreamer_dir = run_mini_fm("vlaps_dreamer_robotfrac1.0_chunk1_wrist", config="tests/mini-dreamer-testing")
        self.assertEqual(code, 0, msg=f"mini_train.py (mini-Dreamer) failed with code {code}. Output:\n{output[-5000:]}")
        code, output, run_dir = run_mini_fm("vlaps_sims2_children3_rollout2", f"init_model={dreamer_dir / 'miniDreamer_best.pth'}",
                                            "sim=testing", "tasks=[0]", "init_states=[0,1]", "vlaps.simulations=2",
                                            "vlaps.children=3", "vlaps.rollout_chunks=2", config="mini-vlaps")
        self.assertEqual(code, 0, msg=f"mini_train.py (mini-VLAPS) failed with code {code}. Output:\n{output[-5000:]}")
        self.assertRegex(output, r"task 0 init state 0: success (True|False), \d+ searches", msg=output[-5000:])
        self.assertRegex(output, r"VLAPS success rate \d", msg=output[-5000:])
        self.assertTrue(glob.glob(str(run_dir / "eval" / "task_0_init_0.mp4")), msg=f"No episode video in {run_dir}")


if __name__ == "__main__":
    unittest.main()

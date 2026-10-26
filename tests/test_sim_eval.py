"""Tests of which task sets the sim evals during training run (sim_eval.py's eval_task_sets): every task set listed
under sim.simulators in the config, or sim.task_sets when it is given, whatever the datasets trained on.

The first tests check the selection on configs, with no sim (a second). test_sim_eval_runs_only_listed_task_sets trains
mini-FM briefly (conf/tests/mini-fm-testing.yaml, whose sim config is conf/sim/testing.yaml) on
two LIBERO suites, and checks that the sim eval runs, and saves videos for, only the one task set that config lists.
"""
import glob
import os
import re
import sys
import unittest
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # mini-wm, for sim_eval.py
from sim_eval import eval_task_sets, maniskill_envs, MANISKILL_TASKS
from test_mini_fm_smoke import REPO, run_mini_fm

LIBERO_SUITES = {"libero_spatial", "libero_goal", "libero_10", "libero_90", "libero_object"}
SIM_CONFIGS = REPO / "conf" / "sim"  # the sim config group: default.yaml, testing.yaml, ...


def sim_file(path):
    """A sim config file of the group, under sim as a run's config has it."""
    return OmegaConf.create({"sim": OmegaConf.load(path)})


def sim_config(libero, maniskill, task_sets=None):
    """A config with the given task sets listed under sim.simulators (as in conf/sim/full.yaml), trained on
    datasets of other task sets."""
    return OmegaConf.create({
        "dataset": {"robot_dataset_name": ["gberseth/libero_10_no_noops_lerobot", "gberseth/maniskill_panda_lerobot"]},
        "sim": {"task_sets": task_sets, "simulators": {
            "libero": {"tasks_per_eval": 10, "task_sets": {t: {"robot": "Franka Panda"} for t in libero} or None},
            "maniskill": {"episodes": 4, "task_sets": {t: {"robot": "WidowX 250s"} for t in maniskill} or None},
            "none": {"datasets": ["gberseth/maniskill_pushletter_lerobot"]}}}})


class TestSimEval(unittest.TestCase):
    def test_eval_task_sets(self):
        """The task sets listed under sim.simulators, in order, and only those: not the datasets'."""
        self.assertEqual(eval_task_sets(sim_config(["libero_90"], ["bridge"])), ["libero_90", "bridge"])
        self.assertEqual(eval_task_sets(sim_config(["libero_spatial", "libero_goal"], [])), ["libero_spatial", "libero_goal"])
        self.assertEqual(eval_task_sets(sim_config([], ["bridge"])), ["bridge"])
        ## sim.task_sets overrides the list
        self.assertEqual(eval_task_sets(sim_config(["libero_90"], ["bridge"], task_sets=["libero_spatial"])), ["libero_spatial"])

    def test_sim_configs_task_sets_exist(self):
        """Every sim config lists at least one task set, and only ones the sim evals can run (libero90.yaml, only an
        alias of default.yaml, lists none of its own)."""
        for path in sorted(SIM_CONFIGS.glob("*.yaml")):
            if "simulators" not in OmegaConf.load(path):
                continue
            task_sets = eval_task_sets(sim_file(path))
            self.assertTrue(task_sets, msg=f"No task sets listed under simulators in {path}")
            for t in task_sets:
                self.assertIn(t, LIBERO_SUITES | set(MANISKILL_TASKS), msg=f"Unknown task set {t} in {path}")

    def test_maniskill_envs(self):
        """The envs a ManiSkill task set's evals run come from its tasks list in the sim config (full.yaml's bridge
        leaves out PutEggplantInBasket), or are all of the task set's without one (testing.yaml)."""
        default = sim_file(SIM_CONFIGS / "full.yaml")
        self.assertEqual(list(maniskill_envs(default, "bridge")), ["PutCarrotOnPlateInScene-v1", "PutSpoonOnTableClothInScene-v1",
                                                                   "StackGreenCubeOnYellowCubeBakedTexInScene-v1"])
        self.assertEqual(list(maniskill_envs(sim_file(SIM_CONFIGS / "testing.yaml"), "bridge")), list(MANISKILL_TASKS["bridge"]))
        default.sim.simulators.maniskill.task_sets.bridge.tasks = ["PutCarrotOnPlateInScene-v1", "NotAnEnv-v1"]
        with self.assertRaises(AssertionError): ## a typo must not silently skip an env
            maniskill_envs(default, "bridge")

    def test_bridge_configs(self):
        """dataset=bridge sim=bridge trains on the WidowX data only, with one camera, and evaluates in the Bridge twins only."""
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(REPO / "conf"), version_base=None):
            cfg = compose(config_name="mini-fm", overrides=["dataset=bridge", "sim=bridge"])
        self.assertEqual(eval_task_sets(cfg), ["bridge"])
        self.assertEqual(len(maniskill_envs(cfg, "bridge")), 3)
        self.assertEqual(len(cfg.dataset.robot_dataset_name), len(cfg.dataset.robot_dataset_weights))
        self.assertEqual(list(cfg.dataset.robot_dataset_name), ["gberseth/bridge_orig_lerobot", "gberseth/maniskill_pushletter_lerobot",
            "gberseth/maniskill_putspoonontablecloth_lerobot", "gberseth/maniskill_putcarrotonplate_lerobot", "gberseth/maniskill_stackcube_lerobot"])
        self.assertEqual((cfg.dataset.cameras, cfg.image_block_size), (None, 64))

    def test_testing_config_task_sets(self):
        """The tests' sim config lists only libero_spatial, the suite they train on."""
        self.assertEqual(eval_task_sets(sim_file(SIM_CONFIGS / "testing.yaml")), ["libero_spatial"])

    def test_sim_eval_runs_only_listed_task_sets(self):
        """Trains on libero_spatial and libero_goal with the tests' sim config, which lists only libero_spatial: the sim
        eval runs libero_spatial alone, not the task set of every dataset trained on."""
        code, output, run_dir = run_mini_fm(
            "fm_simeval_testingsim_train_spatial_goal",
            "dataset.robot_dataset_name=[gberseth/libero_spatial_no_noops_lerobot,gberseth/libero_goal_no_noops_lerobot]")
        self.assertEqual(code, 0, msg=f"mini_train.py failed with code {code}. Output:\n{output[-5000:]}")
        lines = re.findall(r"sim success rate (.*)", output)
        self.assertTrue(lines, msg=f"No sim eval. Output:\n{output[-5000:]}")
        for line in lines:
            evaluated = re.findall(r"(\S+) [\d.]+ \(tasks", line)
            self.assertEqual(evaluated, ["libero_spatial"], msg=f"Evaluated {evaluated}: {line}")
        folders = sorted(os.path.basename(p) for p in glob.glob(str(run_dir / "sim_eval" / "*")))
        self.assertEqual(folders, ["libero_spatial"], msg=f"Videos for {folders} in {run_dir}/sim_eval")
        self.assertTrue(glob.glob(str(run_dir / "sim_eval" / "libero_spatial" / "*.mp4")), msg=f"No video in {run_dir}/sim_eval/libero_spatial")

if __name__ == "__main__":
    unittest.main()

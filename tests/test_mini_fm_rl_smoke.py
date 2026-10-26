import math
import os
import re
import subprocess
import sys
import time
import unittest
from pathlib import Path

"""Smoke tests for mini_fm_rl.py (run by mini_train.py), RL fine-tuning (GRPO or PPO) of a pretrained mini-FM policy in LIBERO.

The tests start from the shipped checkpoint of conf/tests/mini-fm-rl-testing.yaml (init_model, a mini-FM trained on libero_90).
1) mini_train.py with the short RL config (conf/tests/mini-fm-rl-testing.yaml), with GRPO and with PPO: an eval, two
   iterations of one group of 4 short rollouts from sim_eval.py's rollout_libero, a final eval and a saved
   miniFM_rl.pth. GRPO only updates on a group with mixed outcomes; PPO updates every time.
2) One policy_update step on made-up rollouts with advantages +1 and -1 (and with PPO's value head): a finite loss,
   and the weights change.
3) sim_eval.py's envs reset softly (keeping the MuJoCo sim) so they can stay open between episodes: an episode after
   a soft reset must match a fresh env's exactly.
These are pipeline checks, not learning checks. Every run is kept in outputs/tests/<name>_<time>/.
"""

REPO = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "MUJOCO_GL": "egl"}


def run(script, name, *args):
    run_dir = REPO / "outputs" / "tests" / f"{name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    proc = subprocess.run([sys.executable, script, *args, f"hydra.run.dir={run_dir}"], cwd=REPO,
                          capture_output=True, text=True, timeout=3600, env=ENV)
    return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or ""), run_dir


class TestMiniFMRLSmoke(unittest.TestCase):
    def test_grpo_run(self):
        """mini_fm_rl.py with GRPO: evals, two iterations of rollouts and a saved fine-tuned model."""
        code, output, run_dir = run("mini_train.py", "grpo_fm_libero90_task0",
                                    "--config-name", "tests/mini-fm-rl-testing")
        self.assertEqual(code, 0, msg=f"mini_train.py (RL) failed with code {code}. Output:\n{output[-5000:]}")
        self.assertEqual(len(re.findall(r"iter \d+: eval success [\d.]+", output)), 2, msg=output[-5000:])
        used = [float(u) for u in re.findall(r"used_queries ([\d.]+)", output)]
        self.assertEqual(len(used), 2, msg=output[-5000:])
        self.assertTrue(all(math.isfinite(float(l)) for l in re.findall(r", loss ([-\d.naif]+)", output)), msg=output[-5000:])
        self.assertTrue((run_dir / "miniFM_rl.pth").exists())

    def test_ppo_run(self):
        """mini_fm_rl.py with PPO: evals, an update of the policy and the value head every iteration, a saved model."""
        code, output, run_dir = run("mini_train.py", "ppo_fm_libero90_task0",
                                    "--config-name", "tests/mini-fm-rl-testing", "algorithm=ppo")
        self.assertEqual(code, 0, msg=f"mini_train.py (RL) failed with code {code}. Output:\n{output[-5000:]}")
        self.assertEqual(len(re.findall(r"iter \d+: eval success [\d.]+", output)), 2, msg=output[-5000:])
        value_losses = [float(v) for v in re.findall(r"value_loss ([-\d.naif]+)", output)]
        self.assertEqual(len(value_losses), 2, msg=output[-5000:])
        self.assertTrue(all(math.isfinite(v) for v in value_losses), msg=output[-5000:])
        self.assertTrue((run_dir / "miniFM_rl.pth").exists())

    def test_policy_update(self):
        """policy_update on made-up rollouts, with GRPO's advantages and with PPO's value head: a finite loss, and the
        weights change."""
        import torch
        from omegaconf import OmegaConf
        sys.path.insert(0, str(REPO))
        from mini_fm_rl import load_pretrained, policy_update, make_value_head
        from data_loading import n_cameras
        cfg = OmegaConf.load(REPO / "conf" / "mini-fm-rl.yaml")
        model, model_cfg = load_pretrained(str(REPO / cfg.init_model)) ## the shipped mini-FM
        model_cfg.device = cfg.device
        model.to(cfg.device).eval()
        n, episodes = 12, torch.arange(4).repeat_interleave(3) ## 4 episodes of 3 queries
        height, width, channels = model_cfg.image_shape
        queries = {"obs": torch.rand(n, height, width * n_cameras(model_cfg), channels, device=cfg.device), ## the cameras side by side
                   "txt": torch.randint(model_cfg.vocab_size, (n, model_cfg.max_block_size), device=cfg.device),
                   "pose": None, "action": torch.randn(n, model_cfg.action_dim * model_cfg.policy.action_stacking, device=cfg.device),
                   "episode": episodes}
        advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])[episodes] ## of each query: its episode's
        autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        value_head = make_value_head(model, model_cfg)
        copy = lambda net: [p.detach().clone() for p in net.parameters()]
        changed = lambda net, before: any((p != b).any() for p, b in zip(net.parameters(), before))
        ppo = {"value_head": value_head, "value_optimizer": torch.optim.AdamW(value_head.parameters(), lr=1e-3), "returns": torch.rand(n)}
        for value_args in ({}, ppo): ## GRPO, then PPO
            model_before, value_before = copy(model), copy(value_head)
            stats = policy_update(model, torch.optim.AdamW(model.parameters(), lr=1e-3), queries, advantages, cfg, autocast, **value_args)
            self.assertEqual(stats["used_queries"], 1.0)
            self.assertTrue(math.isfinite(stats["loss"]), msg=stats)
            self.assertTrue(changed(model, model_before))
            self.assertEqual(changed(value_head, value_before), bool(value_args))
            if value_args:
                self.assertTrue(math.isfinite(stats["value_loss"]), msg=stats)

    def test_soft_reset_matches_fresh_env(self):
        """An episode in a LiberoChunkEnv that already ran another one matches a fresh env's, state for state."""
        import numpy as np
        os.environ.setdefault("MUJOCO_GL", "egl")
        sys.path.insert(0, str(REPO))
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from sim_eval import LiberoChunkEnv
        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        task, states = suite.get_task(0), suite.get_task_init_states(0)
        actions = np.clip(np.random.default_rng(0).normal(0, 0.5, (40, 7)), -1, 1)
        actions[:, 6] = np.where(np.arange(40) < 20, 1.0, -1.0) ## close, then open the gripper
        def episode(env, state, chunks=True): ## a LiberoChunkEnv steps chunks, here of one step each
            env.reset()
            env.set_init_state(state)
            return np.stack([env.step(a[None] if chunks else a)[0] and env.get_sim_state() for a in actions])
        fresh = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                                   camera_heights=64, camera_widths=64)
        fresh.seed(0)
        soft = LiberoChunkEnv(task, 64)
        episode(soft, states[7]) ## another episode first, which ends with the gripper open
        np.testing.assert_array_equal(episode(soft, states[3]), episode(fresh, states[3], chunks=False))
        fresh.close(); soft.close()


if __name__ == "__main__":
    unittest.main()

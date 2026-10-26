# mini-wm's tests

Short smoke runs of each model: a few training steps on LIBERO spatial, then one short sim episode. They check that
the code runs, not that it learns. Run them from mini-wm's folder (`uv run -m pytest tests/ -v`), or from the repo
that holds it (`uv run -m pytest mini-wm/tests/ -v`). The ManiSkill and Bridge ManiSkill tests are skipped without
ManiSkill or the Bridge assets.

## mini-FM smoke tests

`test_mini_fm_smoke.py` checks `mini_fm.py`'s models, trained by `mini_train.py`, for each `model.type` and action head:
- `fm`, the transformer, with the Huber action head (the default) and with the MSE, L1 and diffusion heads (`model.action_head=mse|l1|diffusion`);
- `fm` with one class token per action of the chunk (`model.token_per_action=true`), each decoded by the same linear layer;
- `convnet`, `ConvNetFM` (mini-GRP's convnet in `mini_fm.py`), conditioned on the goal text with no goal image, with the MSE action head and with the diffusion head.
- `fm` with the sim eval in a background process (`sim.background=true`, as in real runs); the others run it in-process.
- `fm` with the background sim eval, killing a LIBERO env process during the first eval: the evaluator must restart and the run still end with a sim eval (`background_sim_eval_restart`, about 2 min).
- `fm` and `convnet` given the robot's pose, its arm joint positions (`policy.use_pose_data=true`).
- `fm` on LIBERO spatial and ManiSkill PushT data together (`gberseth/maniskill_pusht_lerobot`, made with `openvla` `experiments/robot/maniskill/regenerate_maniskill_dataset.py`). The sim eval runs in both, LIBERO spatial and ManiSkill `PushT-v1`.
- `fm` on LIBERO spatial and the ManiSkill Panda-stick tasks (`gberseth/maniskill_panda_stick_lerobot`: PushT, DrawTriangle), with sim evals in LIBERO spatial and both ManiSkill tabletop task sets, `maniskill_panda_stick` and `maniskill_panda` (11 tasks of the Panda with its gripper). The small stick dataset stands in for `gberseth/maniskill_panda_lerobot` (711k frames) to keep the download small.
- `fm` on the agent view (`dataset.robot_image_key=observation.images.image`), with sim evals in LIBERO spatial and Bridge ManiSkill (`sim.task_sets=[libero_spatial,bridge]`, the WidowX scenes where models trained on `gberseth/bridge_orig_lerobot` are evaluated). It trains on LIBERO spatial only, to keep the download small. It needs Bridge ManiSkill's assets (`python -m mani_skill.utils.download_asset bridge_v2_real2sim`, and `widowx250s`, into `$MS_ASSET_DIR`, by default `~/.maniskill`).
- `fm` on LIBERO spatial (wrist camera, 7 joints) and ManiSkill's WidowX PushLetter (`gberseth/maniskill_pushletter_lerobot`, agent view, 6 joints zero-padded to 7) together, given the pose (`policy.use_pose_data=true`). PushLetter's env isn't in ManiSkill, so only LIBERO spatial is evaluated.
- `fm` on LIBERO spatial (wrist camera) and the demos recorded in Bridge ManiSkill's PutSpoon (`gberseth/maniskill_putspoonontablecloth_lerobot`, agent view, 5 Hz), with sim evals in LIBERO spatial and that task (task set `PutSpoonOnTableClothInScene-v1`), whose actions go to the env unconverted.

Each test does three things:
- trains for 300 steps on libero_spatial robot data (the short run of `conf/tests/mini-fm-testing.yaml`; the tests add only the model overrides);
- runs one short in-process sim eval;
- checks for finite val action-head losses (mse or diffusion), a sim success rate, a saved rollout MP4, and a saved best model (`miniFM_best.pth`).

Each run is kept in `outputs/tests/<name>_<time>/`. The tests check the pipeline, not success.

After each code change, run only the quick check: `test_mini_fm_fm_robot_smoke` plus any new test or test that covers the change (about a minute each). Run the whole file at the end of the day.

```bash
uv run -m pytest tests/test_mini_fm_smoke.py -v -k "fm_robot_smoke"  # after each change, e.g. -k "fm_robot_smoke or pose"
uv run -m pytest tests/test_mini_fm_smoke.py -v                      # at the end of the day
```

## Sim eval task sets

`test_sim_eval.py` checks which task sets the sim evals during training run (`sim_eval.py`'s `eval_task_sets`): every task set listed under `sim.simulators` in the sim config (`conf/sim/`: `default.yaml` for training, `testing.yaml` for the tests), or `sim.task_sets` when given, never the datasets'. `test_maniskill_envs` checks that a ManiSkill task set's envs come from its `tasks` list in the sim config (`default.yaml`'s `bridge` leaves out PutEggplantInBasket), all of them without one, and that an unknown env id is an error. `test_bridge_configs` checks that `dataset=bridge sim=bridge` trains on the WidowX datasets only, with one camera, and evaluates only in Bridge ManiSkill. Its config tests take a second; `test_sim_eval_runs_only_listed_task_sets` trains briefly on two LIBERO suites and checks the eval runs only the one `testing.yaml` lists.

```bash
uv run -m pytest tests/test_sim_eval.py -v
```

## mini-WAM smoke test

`test_mini_wam_smoke.py` checks `mini_wam.py`, the world action model, with the short run of `conf/tests/mini-wam-testing.yaml` (that of `mini-fm-testing.yaml` with `model.type: wam`). It checks for finite val losses of the action head (`mse`) and of the frame diffusion (`frames`), the frame loss by dataset, a saved grid of predicted frames (`predicted_frames/*.png`), a sim success rate, a rollout MP4 and `miniWAM_best.pth`. Run it after changes to `mini_wam.py`, `mini_train.py` or the robot cache's future frames.

```bash
uv run -m pytest tests/test_mini_wam_smoke.py -v  # about 2 min
```

## RL fine-tuning smoke tests (GRPO and PPO)

`test_mini_fm_rl_smoke.py` checks `mini_fm_rl.py` (run by `mini_train.py`), RL fine-tuning (GRPO or PPO) of a pretrained mini-FM policy in LIBERO. The tests start from the shipped checkpoint, `checkpoints/miniFM_libero90.pth` (`init_model`), except the soft-reset test. GRPO only updates on a group of rollouts with mixed outcomes.
- `test_grpo_run` runs the short config `conf/tests/mini-fm-rl-testing.yaml`: an eval, two iterations of one group of 4 short rollouts on libero_90 task 0, and a final eval. It checks that both iterations ran and that `miniFM_rl.pth` was saved.
- `test_ppo_run` runs the same short config with `algorithm=ppo`. It checks for two evals, a finite value-head loss in both iterations, and a saved `miniFM_rl.pth`.
- `test_policy_update` runs one `policy_update` step on made-up rollouts with advantages +1 and -1, once as GRPO and once with PPO's value head. It checks for a finite loss, changed policy weights, and value-head weights that change only with PPO.
- `test_soft_reset_matches_fresh_env` checks that `sim_eval.py`'s envs, which reset softly so they can stay open between episodes, run an episode exactly like a fresh env, state for state, after already running another one.

```bash
uv run -m pytest tests/test_mini_fm_rl_smoke.py -v  # about 2 min 40 s
```

`test_mini_vlaps_smoke.py` checks `mini_vlaps.py` (run by `mini_train.py --config-name mini-vlaps`), Monte Carlo tree search with a pre-trained mini-Dreamer and its mini-FM, as VLAPS. It trains the short mini-Dreamer of `conf/tests/mini-dreamer-testing.yaml`, then runs a small search (2 simulations of 3 children) for one short episode of libero_spatial task 0, and checks that the episode's result, the success rate and its video are there.

```bash
uv run -m pytest tests/test_mini_vlaps_smoke.py -v
```


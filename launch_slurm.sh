#!/bin/bash
## One Slurm job per seed (the array index is r_seed). Submit from the repo root, e.g.
##   mkdir -p outputs/slurm && sbatch --array=1-2 --job-name=fm_chunk5_wristagentview launch_slurm.sh
##   mkdir -p outputs/slurm && sbatch --array=1-2 --job-name=wam_chunk5_wrist launch_slurm.sh --config-name mini-wam
## Overrides go after the script name. The job name becomes the wandb group (each seed logs as <name>_seed<n>), so name
## it after the settings the experiment changes. Each seed runs in outputs/slurm/<job name>/<array job id>_<seed>/; the
## Slurm log is outputs/slurm/<job name>-<array job id>_<seed>.out. Adapt the #SBATCH lines to your cluster.

#SBATCH --cpus-per-task=8                             # the LIBERO sim evals run one env process per task
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                                  # bf16 needs an Ampere or newer GPU (or pass precision=fp32)
#SBATCH --mem=64G                                     # the dataset cache is in RAM (dataset.cache: cpu): about 14 GB for LIBERO-90 with two cameras
#SBATCH --time=1:30:00                                # a 25k-step mini-FM run: a few minutes of loading, about 25 min of training, then the sim evals
#SBATCH -o outputs/slurm/%x-%A_%a.out                 # relative to the submit dir; outputs/slurm must exist

cd "$SLURM_SUBMIT_DIR"
export MUJOCO_GL=egl ## the LIBERO sim evals render on the GPU
RUN_DIR=outputs/slurm/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
echo "Starting mini_train.py on $SLURM_JOB_NODELIST, seed $SLURM_ARRAY_TASK_ID, run dir $RUN_DIR, args: $*"
## The arguments come first: hydra needs options such as --config-name before the key=value overrides.
## testing=false turns wandb on (log in once with `uv run wandb login`).
uv run mini_train.py "$@" r_seed=$SLURM_ARRAY_TASK_ID testing=false experiment.name=${SLURM_JOB_NAME} hydra.run.dir=$RUN_DIR

## One logger for training runs: Weights & Biases (unless cfg.testing) and/or TensorBoard (cfg.tensorboard).
import os

class Logger:
    """ Logs to wandb and/or TensorBoard, depending on the config. wandb groups the runs of one experiment
    (e.g. the seeds of a Slurm array) under experiment.name, so it can average them. TensorBoard writes to
    ./tensorboard in the run folder (hydra runs inside outputs/<date>/<time>/); view it with tensorboard --logdir outputs. """

    def __init__(self, cfg):
        from omegaconf import OmegaConf
        self.wandb = self.tb = None
        if not cfg.testing:
            import wandb
            ## wandb limits group and run names to 128 characters: the group is cut to 121, so the run name <group>_seed<n>
            ## fits too, and the config keeps the whole name in experiment_full_name
            group = cfg.experiment.name[:121]
            ## the Slurm job id (the array job's for launch_slurm.sh), so a job's runs can be found by id in wandb
            slurm_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID"))
            wandb.init(entity=cfg.experiment.get("entity"), project=cfg.experiment.project, group=group, name=f"{group}_seed{cfg.r_seed}",
                       config={**OmegaConf.to_container(cfg), "experiment_full_name": cfg.experiment.name,
                               "slurm_job_id": slurm_job_id})
            wandb.run.log_code(".")
            ## Sim evals finish late (in a background process), after training has logged later steps, so every metric
            ## named sim... gets its own step axis, sim/step
            wandb.define_metric("sim/step")
            wandb.define_metric("sim/*", step_metric="sim/step")
            wandb.define_metric("sim_*", step_metric="sim/step") ## sim_distance/, sim_<suite>/ and sim_video_<suite>/
            self.wandb = wandb
        if cfg.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter("tensorboard")

    def log(self, scalars, step, tensorboard_only={}, videos={}, images={}):
        """ Logs {name: value} at step. Names starting with "sim" go on wandb's sim/step axis. tensorboard_only are
        scalars too many to plot in wandb; videos ({name: MP4 path}) go to wandb only, since TensorBoard would store
        them as GIFs about 50x larger. images: {name: (H, W, 3) uint8 array}, to both. """
        if self.tb is not None:
            for name, value in {**scalars, **tensorboard_only}.items():
                self.tb.add_scalar(name, value, step)
            for name, image in images.items():
                self.tb.add_image(name, image, step, dataformats="HWC")
            self.tb.flush()
        if self.wandb is not None:
            videos = {name: self.wandb.Video(path, format="mp4") for name, path in videos.items()}
            images = {name: self.wandb.Image(image) for name, image in images.items()}
            if all(name.startswith("sim") for name in scalars):
                self.wandb.log({"sim/step": step, **scalars, **videos, **images})
            else:
                self.wandb.log({**scalars, **videos, **images}, step=step)

    def close(self):
        if self.tb is not None:
            self.tb.close()
        if self.wandb is not None:
            self.wandb.finish()

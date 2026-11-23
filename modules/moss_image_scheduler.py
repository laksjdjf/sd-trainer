from modules.scheduler import FlowScheduler


class MOSSImageTrainingScheduler(FlowScheduler):
    def __init__(self, sampling_scheduler):
        super().__init__(shift=sampling_scheduler.config.training_shift)
        self.sampling_scheduler = sampling_scheduler

    def set_timesteps(self, num_inference_steps, device="cuda"):
        self.num_inference_steps = num_inference_steps
        self.sampling_scheduler.set_timesteps(num_inference_steps, device=device)
        return self.sampling_scheduler.timesteps

import numpy as np
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
from modules.scheduler import FlowScheduler


class QwenImage21Scheduler(FlowScheduler):
    """Flow loss with resolution-dependent shifting and official sampling timesteps."""

    def __init__(self, scheduler):
        super().__init__(shift=1.0)
        self.diffusers_scheduler = scheduler
        self.configure_latents((1, 64, 1, 64, 64))

    def configure_latents(self, shape):
        config = self.diffusers_scheduler.config
        self.mu = calculate_shift(
            shape[-2] * shape[-1], config.base_image_seq_len,
            config.max_image_seq_len, config.base_shift, config.max_shift,
        )
        self.shift = float(np.exp(self.mu)) if config.use_dynamic_shifting else config.shift

    def set_timesteps(self, num_inference_steps, device="cuda"):
        self.num_inference_steps = num_inference_steps
        self.diffusers_scheduler.set_timesteps(
            sigmas=np.linspace(1.0, 1 / num_inference_steps, num_inference_steps),
            device=device, mu=self.mu,
        )
        return self.diffusers_scheduler.timesteps

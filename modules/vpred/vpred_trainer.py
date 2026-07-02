import torch
from modules.trainer import BaseTrainer

class VPredTrainer(BaseTrainer):
    def __init__(self, config, model_type, diffusion, text_model, vae, diffusers_scheduler, scheduler, network, nf4=False, taesd=False):
        super().__init__(config, model_type, diffusion, text_model, vae, diffusers_scheduler, scheduler, network, nf4, taesd)
        self.scheduler.v_prediction = True
        self.scheduler.zsnr = True

    def loss(self, batch):
        latents = self.latents_from_batch(batch)
        self.batch_size = latents.shape[0] # stepメソッドでも使う
        text_output = self.text_output_from_batch(batch)

        if "size_condition" in batch:
            size_condition = batch["size_condition"].to(self.device)
        else:
            size_condition = None

        timesteps = self.scheduler.sample_timesteps(latents.shape[0], self.device)
        noise = self.sample_noise(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            model_output_student = self.diffusion(noisy_latents, timesteps, text_output, sample=False, size_condition=size_condition)
            with self.network.set_temporary_multiplier(0.0), torch.no_grad():
                model_output_teacher = self.diffusion(noisy_latents, timesteps, text_output, sample=False, size_condition=size_condition)

        target = self.scheduler.get_target(latents, model_output_teacher, timesteps)

        loss = torch.nn.functional.mse_loss(model_output_student.float(), target.float(), reduction="mean")

        return loss
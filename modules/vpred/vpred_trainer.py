import torch
from modules.text_model import BaseTextOutput
from modules.trainer import BaseTrainer
from modules.scheduler import FlowScheduler

class VPredTrainer(BaseTrainer):
    def __init__(self, config, model_type, diffusion, text_model, vae, diffusers_scheduler, scheduler, network, nf4=False, taesd=False):
        super().__init__(config, model_type, diffusion, text_model, vae, diffusers_scheduler, scheduler, network, nf4, taesd)
        self.scheduler.v_prediction = True
        self.scheduler.zsnr = True
    
    def loss(self, batch):
        if "latents" in batch:
            latents = batch["latents"].to(self.device)
        else:
            with torch.autocast("cuda", dtype=self.vae_dtype), torch.no_grad():
                latents = self.vae.encode(batch['images'].to(self.device)).latent_dist.sample()
        latents = (latents - self.shift_factor) * self.scaling_factor
        
        self.batch_size = latents.shape[0] # stepメソッドでも使う

        if "encoder_hidden_states" in batch:
            encoder_hidden_states = batch["encoder_hidden_states"].to(self.device)
            pooled_output = batch["pooled_outputs"].to(self.device)
            text_output = BaseTextOutput(encoder_hidden_states, pooled_output)
        else:
            with torch.autocast("cuda", dtype=self.autocast_dtype):
                text_output = self.text_model(batch["captions"])

        if "size_condition" in batch:
            size_condition = batch["size_condition"].to(self.device)
        else:
            size_condition = None

        timesteps = self.scheduler.sample_timesteps(latents.shape[0], self.device)
        noise = torch.randn_like(latents)
        if self.config.noise_offset != 0:
            noise += self.config.noise_offset * torch.randn(noise.shape[0], noise.shape[1], 1, 1).to(noise)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            model_output_student = self.diffusion(noisy_latents, timesteps, text_output, sample=False, size_condition=size_condition)
            with self.network.set_temporary_multiplier(0.0), torch.no_grad():
                model_output_teacher = self.diffusion(noisy_latents, timesteps, text_output, sample=False, size_condition=size_condition)

        target = self.scheduler.get_target(latents, model_output_teacher, timesteps)

        loss = torch.nn.functional.mse_loss(model_output_student.float(), target.float(), reduction="mean")

        return loss
    
class FlowTrainer(BaseTrainer):
    def __init__(self, config, model_type, diffusion, text_model, vae, diffusers_scheduler, scheduler, network, nf4=False, taesd=False):
        super().__init__(config, model_type, diffusion, text_model, vae, diffusers_scheduler, FlowScheduler(), network, nf4, taesd)
    
class CFGDistillTrainer(BaseTrainer):
    def loss(self, batch):
        if "latents" in batch:
            latents = batch["latents"].to(self.device)
        else:
            with torch.autocast("cuda", dtype=self.vae_dtype), torch.no_grad():
                latents = self.vae.encode(batch['images'].to(self.device)).latent_dist.sample()
        latents = (latents - self.shift_factor) * self.scaling_factor
        
        self.batch_size = latents.shape[0] # stepメソッドでも使う

        if "encoder_hidden_states" in batch:
            encoder_hidden_states = batch["encoder_hidden_states"].to(self.device)
            pooled_output = batch["pooled_outputs"].to(self.device)
            text_output = BaseTextOutput(encoder_hidden_states, pooled_output)
        else:
            with torch.autocast("cuda", dtype=self.autocast_dtype):
                text_output = self.text_model(batch["captions"])

        if "size_condition" in batch:
            size_condition = batch["size_condition"].to(self.device)
        else:
            size_condition = None

        timesteps = self.scheduler.sample_timesteps(latents.shape[0], self.device)
        noise = torch.randn_like(latents)
        if self.config.noise_offset != 0:
            noise += self.config.noise_offset * torch.randn(noise.shape[0], noise.shape[1], 1, 1).to(noise)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            model_output_student = self.diffusion(noisy_latents, timesteps, text_output, sample=False, size_condition=size_condition)
            with self.network.set_temporary_multiplier(0.0), torch.no_grad():
                negative_output = self.text_model.negative_output.repeat(self.batch_size)
                model_output_positive = self.diffusion(noisy_latents, timesteps, text_output, sample=False, size_condition=size_condition)
                model_output_negative = self.diffusion(noisy_latents, timesteps, negative_output, sample=False, size_condition=size_condition)
                model_output_teacher = model_output_negative + 7.0 * (model_output_positive - model_output_negative)

        loss = torch.nn.functional.mse_loss(model_output_student.float(), model_output_teacher.float(), reduction="mean")

        return loss
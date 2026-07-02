import torch
import torch.nn as nn
import torch.nn.functional
from modules.trainer import BaseTrainer
from modules.text_model import BaseTextOutput

class SliderTrainer(BaseTrainer):
    def loss(self, batch):
        if "latents_w" in batch:
            latents_w = batch["latents_w"].to(self.device)
            latents_l = batch["latents_l"].to(self.device)
        else:
            with torch.autocast("cuda", dtype=self.vae_dtype), torch.no_grad():
                latents_w = self.vae.encode(batch['images_w'].to(self.device)).latent_dist.sample()
                latents_l = self.vae.encode(batch['images_l'].to(self.device)).latent_dist.sample()
        latents_w = (latents_w - self.shift_factor) * self.scaling_factor
        latents_l = (latents_l - self.shift_factor) * self.scaling_factor
        
        self.batch_size = latents_w.shape[0] # stepメソッドでも使う

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

        timesteps = self.scheduler.sample_timesteps(latents_w.shape[0], self.device)
        noise = torch.randn_like(latents_w)
        if self.config.noise_offset != 0:
            noise += self.config.noise_offset * torch.randn(noise.shape[0], noise.shape[1], 1, 1).to(noise)
        noisy_latents_w = self.scheduler.add_noise(latents_w, noise, timesteps)
        noisy_latents_l = self.scheduler.add_noise(latents_l, noise, timesteps)
        
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            model_output_w = self.diffusion(noisy_latents_w, timesteps, text_output, sample=False, size_condition=size_condition)
            with self.network.set_temporary_multiplier(-1.0):
                model_output_l = self.diffusion(noisy_latents_l, timesteps, text_output, sample=False, size_condition=size_condition)
        
        target_w = self.scheduler.get_target(latents_w, noise, timesteps)
        target_l = self.scheduler.get_target(latents_l, noise, timesteps)
        
        loss_w = nn.functional.mse_loss(model_output_w.float(), target_w.float(), reduction="mean")
        loss_l = nn.functional.mse_loss(model_output_l.float(), target_l.float(), reduction="mean")
        
        return loss_w + loss_l

class ADDifTTrainer(BaseTrainer):
    def loss(self, batch):
        if not hasattr(self, "alternate"):
            self.alternate = 1.0

        if "latents_w" in batch:
            latents_w = batch["latents_w"].to(self.device)
            latents_l = batch["latents_l"].to(self.device)
        else:
            with torch.autocast("cuda", dtype=self.vae_dtype), torch.no_grad():
                latents_w = self.vae.encode(batch['images_w'].to(self.device)).latent_dist.sample()
                latents_l = self.vae.encode(batch['images_l'].to(self.device)).latent_dist.sample()
        latents_w = (latents_w - self.shift_factor) * self.scaling_factor
        latents_l = (latents_l - self.shift_factor) * self.scaling_factor

        
        self.batch_size = latents_w.shape[0] # stepメソッドでも使う

        if "encoder_hidden_states" in batch:
            encoder_hidden_states = batch["encoder_hidden_states"].to(self.device)
            pooled_output = batch["pooled_outputs"].to(self.device)
            text_output = BaseTextOutput(encoder_hidden_states, pooled_output)
        else:
            with torch.autocast("cuda", dtype=self.autocast_dtype):
                text_output = self.text_model(batch["captions"])

        if "encoder_hidden_states_l" in batch:
            encoder_hidden_states_l = batch["encoder_hidden_states_l"].to(self.device)
            pooled_output_l = batch["pooled_outputs_l"].to(self.device)
            text_output_l = BaseTextOutput(encoder_hidden_states_l, pooled_output_l)
        else:
            if "captions_l" in batch:
                with torch.autocast("cuda", dtype=self.autocast_dtype):
                    text_output_l = self.text_model(batch["captions_l"])
            else:
                text_output_l = text_output

        if self.alternate == -1.0:
            latents_w, latents_l = latents_l, latents_w
            text_output, text_output_l = text_output_l, text_output

        if "size_condition" in batch:
            size_condition = batch["size_condition"].to(self.device)
        else:
            size_condition = None

        timesteps = self.scheduler.sample_timesteps(latents_w.shape[0], self.device)
        noise = torch.randn_like(latents_w)
        if self.config.noise_offset != 0:
            noise += self.config.noise_offset * torch.randn(noise.shape[0], noise.shape[1], 1, 1).to(noise)
        noisy_latents_w = self.scheduler.add_noise(latents_w, noise, timesteps)
        noisy_latents_l = self.scheduler.add_noise(latents_l, noise, timesteps)
        
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            with self.network.set_temporary_multiplier(self.alternate):
                model_output_w = self.diffusion(noisy_latents_w, timesteps, text_output, sample=False, size_condition=size_condition)
            with self.network.set_temporary_multiplier(0.0), torch.no_grad():
                model_output_l = self.diffusion(noisy_latents_l, timesteps, text_output_l, sample=False, size_condition=size_condition)
        
        loss = nn.functional.mse_loss(model_output_w.float(), model_output_l.float(), reduction="mean")

        self.alternate *= -1
        
        return loss
    
class SelfRegTrainer(BaseTrainer):
    def loss(self, batch):
        if not hasattr(self, "alternate"):
            self.alternate = 1.0

        if "latents_w" in batch:
            latents_w = batch["latents_w"].to(self.device)
            latents_l = batch["latents_l"].to(self.device)
        else:
            with torch.autocast("cuda", dtype=self.vae_dtype), torch.no_grad():
                latents_w = self.vae.encode(batch['images_w'].to(self.device)).latent_dist.sample()
                latents_l = self.vae.encode(batch['images_l'].to(self.device)).latent_dist.sample()
        latents_w = (latents_w - self.shift_factor) * self.scaling_factor
        latents_l = (latents_l - self.shift_factor) * self.scaling_factor

        if self.alternate == -1.0:
            latents_w, latents_l = latents_l, latents_w
        
        self.batch_size = latents_w.shape[0] # stepメソッドでも使う

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

        timesteps = self.scheduler.sample_timesteps(latents_w.shape[0], self.device)
        noise = torch.randn_like(latents_w)
        if self.config.noise_offset != 0:
            noise += self.config.noise_offset * torch.randn(noise.shape[0], noise.shape[1], 1, 1).to(noise)
        noisy_latents_w = self.scheduler.add_noise(latents_w, noise, timesteps)
        noisy_latents_l = self.scheduler.add_noise(latents_l, noise, timesteps)
        
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            with self.network.set_temporary_multiplier(self.alternate):
                model_output_w = self.diffusion(noisy_latents_w, timesteps, text_output, sample=False, size_condition=size_condition)
                model_output_l = self.diffusion(noisy_latents_l, timesteps, text_output, sample=False, size_condition=size_condition)
            with self.network.set_temporary_multiplier(0.0), torch.no_grad():
                model_output_reg = self.diffusion(noisy_latents_l, timesteps, text_output, sample=False, size_condition=size_condition)
        
        target_w = self.scheduler.get_target(latents_w, noise, timesteps)

        loss_w = nn.functional.mse_loss(model_output_w.float(), target_w.float(), reduction="mean")
        loss_reg = nn.functional.mse_loss(model_output_l.float(), model_output_reg.float(), reduction="mean")
                
        return loss_w + loss_reg
    
class ADDifT2Trainer(BaseTrainer):
    def loss(self, batch):
        if not hasattr(self, "alternate"):
            self.alternate = 1.0

        if "latents_w" in batch:
            if self.alternate == 1.0:
                latents_w = batch["latents_w"].to(self.device)
            else:
                latents_w = batch["latents_l"].to(self.device)
            latents_n = batch["latents_n"].to(self.device)
        else:
            with torch.autocast("cuda", dtype=self.vae_dtype), torch.no_grad():
                if self.alternate == 1.0:
                    latents_w = self.vae.encode(batch['images_w'].to(self.device)).latent_dist.sample()
                else:
                    latents_w = self.vae.encode(batch['images_l'].to(self.device)).latent_dist.sample()
                latents_n = self.vae.encode(batch['images_n'].to(self.device)).latent_dist.sample()
        latents_w = (latents_w - self.shift_factor) * self.scaling_factor
        latents_n = (latents_n - self.shift_factor) * self.scaling_factor
        
        self.batch_size = latents_w.shape[0] # stepメソッドでも使う

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

        timesteps = self.scheduler.sample_timesteps(latents_w.shape[0], self.device)
        noise = torch.randn_like(latents_w)
        if self.config.noise_offset != 0:
            noise += self.config.noise_offset * torch.randn(noise.shape[0], noise.shape[1], 1, 1).to(noise)
        noisy_latents_w = self.scheduler.add_noise(latents_w, noise, timesteps)
        noisy_latents_n = self.scheduler.add_noise(latents_n, noise, timesteps)
        
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            with self.network.set_temporary_multiplier(self.alternate):
                model_output_w = self.diffusion(noisy_latents_w, timesteps, text_output, sample=False, size_condition=size_condition)
            with self.network.set_temporary_multiplier(0.0), torch.no_grad():
                model_output_n = self.diffusion(noisy_latents_n, timesteps, text_output, sample=False, size_condition=size_condition)
        
        loss = nn.functional.mse_loss(model_output_w.float(), model_output_n.float(), reduction="mean")

        self.alternate *= -1
        
        return loss

class ADDifTPMTrainer(BaseTrainer):
    def loss(self, batch):
        if not hasattr(self, "alternate"):
            self.alternate = 1.0

        if "latents_w" in batch:
            latents_w = batch["latents_w"].to(self.device)
            latents_l = batch["latents_l"].to(self.device)
        else:
            with torch.autocast("cuda", dtype=self.vae_dtype), torch.no_grad():
                latents_w = self.vae.encode(batch['images_w'].to(self.device)).latent_dist.sample()
                latents_l = self.vae.encode(batch['images_l'].to(self.device)).latent_dist.sample()
        latents_w = (latents_w - self.shift_factor) * self.scaling_factor
        latents_l = (latents_l - self.shift_factor) * self.scaling_factor

        
        self.batch_size = latents_w.shape[0] # stepメソッドでも使う

        if "encoder_hidden_states" in batch:
            encoder_hidden_states = batch["encoder_hidden_states"].to(self.device)
            pooled_output = batch["pooled_outputs"].to(self.device)
            text_output = BaseTextOutput(encoder_hidden_states, pooled_output)
        else:
            with torch.autocast("cuda", dtype=self.autocast_dtype):
                text_output = self.text_model(batch["captions"])

        if "encoder_hidden_states_l" in batch:
            encoder_hidden_states_l = batch["encoder_hidden_states_l"].to(self.device)
            pooled_output_l = batch["pooled_outputs_l"].to(self.device)
            text_output_l = BaseTextOutput(encoder_hidden_states_l, pooled_output_l)
        else:
            if "captions_l" in batch:
                with torch.autocast("cuda", dtype=self.autocast_dtype):
                    text_output_l = self.text_model(batch["captions_l"])
            else:
                text_output_l = text_output

        #if self.alternate == -1.0:
        #    latents_w, latents_l = latents_l, latents_w
        #    text_output, text_output_l = text_output_l, text_output

        if "size_condition" in batch:
            size_condition = batch["size_condition"].to(self.device)
        else:
            size_condition = None

        timesteps = self.scheduler.sample_timesteps(latents_w.shape[0], self.device)
        noise = torch.randn_like(latents_w)
        if self.config.noise_offset != 0:
            noise += self.config.noise_offset * torch.randn(noise.shape[0], noise.shape[1], 1, 1).to(noise)
        noisy_latents_w = self.scheduler.add_noise(latents_w, noise, timesteps)
        noisy_latents_l = self.scheduler.add_noise(latents_l, noise, timesteps)
        
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            with self.network.set_temporary_multiplier(self.alternate):
                model_output_w = self.diffusion(noisy_latents_w, timesteps, text_output, sample=False, size_condition=size_condition)
            with self.network.set_temporary_multiplier(0.0), torch.no_grad():
                model_output_l = self.diffusion(noisy_latents_l, timesteps, text_output_l, sample=False, size_condition=size_condition)
        
        loss = nn.functional.mse_loss(model_output_w.float(), model_output_l.float(), reduction="mean")

        self.alternate *= -1
        
        return loss
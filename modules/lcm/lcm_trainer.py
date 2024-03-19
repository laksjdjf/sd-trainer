import torch
import torch.nn as nn
from modules.trainer import BaseTrainer
from modules.scheduler import BaseScheduler, substitution_t

# additional config
# trainer:
#   additional_conf:
#     lcm:
#       negative_prompt: "low quality"
#       guidance_scale: 7.0
#       num_inference_steps: 50

class LCMScheduler(BaseScheduler):
     # x_t -> x_prev_t
    def step(self, sample, model_output, t, prev_t, use_ddim=False):
        pred_original_sample = self.pred_original_sample(sample, model_output, t)
        if use_ddim: # for training step
            noise = self.pred_noise(sample, model_output, t)
        elif self.gamma == 0.0:
            noise = torch.randn_like(sample)
        else: # tcd
            inner_t = ((1 - self.gamma) * prev_t).round().long()
            output_inner = self.step(sample, model_output, t, inner_t, use_ddim=True)
            noise_random = torch.randn_like(sample)
            return self.add_noise_inner(output_inner, noise_random, inner_t, prev_t)

        return self.add_noise(pred_original_sample, noise, prev_t)
    
    # x_current_t -> x_target_t
    def add_noise_inner(self, sample, noise, current_t, target_t):
        alphas_bar_current = substitution_t(self.alphas_bar, current_t, sample.shape[0])
        alphas_bar_target = substitution_t(self.alphas_bar, target_t, sample.shape[0])

        alphas_bar = alphas_bar_target / alphas_bar_current

        return alphas_bar.sqrt() * sample + (1 - alphas_bar).sqrt() * noise
        

class LCMTrainer(BaseTrainer):
    def __init__(self, config, diffusion, text_model, vae, scheduler, network):
        super().__init__(config, diffusion, text_model, vae, scheduler, network)
        self.scheduler = LCMScheduler(self.scheduler.v_prediction) # overwrite
        self.tcd = config.additional_conf.lcm.get("tcd", False)
        gamma = 0.3 if self.tcd else 0.0
        setattr(self.scheduler, "gamma", gamma)

    def prepare_modules_for_training(self, device="cuda"):
        super().prepare_modules_for_training(device)

        self.text_model.to(device)
        self.negative_encoder_hidden_states, self.negative_pooled_output = self.text_model([self.config.additional_conf.lcm.negative_prompt])
        self.text_model.to(self.te_device)

    def loss(self, batch):
        if "latents" in batch:
            latents = batch["latents"].to(self.device) * self.vae.scaling_factor
        else:
            with torch.autocast("cuda", dtype=self.vae_dtype), torch.no_grad():
                latents = self.vae.encode(batch['images'].to(self.device)).latent_dist.sample() * self.vae.scaling_factor
        
        self.batch_size = latents.shape[0] # stepメソッドでも使う

        if "encoder_hidden_states" in batch:
            encoder_hidden_states = batch["encoder_hidden_states"].to(self.device)
            pooled_output = batch["pooled_outputs"].to(self.device)
        else:
            with torch.autocast("cuda", dtype=self.autocast_dtype):
                encoder_hidden_states, pooled_output = self.text_model(batch["captions"])

        if "size_condition" in batch:
            size_condition = batch["size_condition"].to(self.device)
        else:
            size_condition = None

        num_inference_steps = self.config.additional_conf.lcm.num_inference_steps
        interval = 1000 // num_inference_steps
        timesteps = torch.randint(interval, 1000, (self.batch_size,), device=latents.device)
        prev_timesteps = timesteps - interval
        if self.tcd:
            inner_timesteps = []
            for t in prev_timesteps:
                inner_timesteps.append(torch.randint(0, t+1, (1,), device=latents.device))
            inner_timesteps = torch.cat(inner_timesteps)

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            model_output = self.diffusion(noisy_latents, timesteps, encoder_hidden_states, pooled_output, size_condition)
            pred_original_sample = self.scheduler.pred_original_sample(noisy_latents, model_output, timesteps)
            pred = self.scheduler.step(noisy_latents, model_output, timesteps, inner_timesteps, use_ddim=True) if self.tcd else pred_original_sample
            with torch.no_grad():
                # one step ddim
                negative_encoder_hidden_states = self.negative_encoder_hidden_states.repeat(self.batch_size, 1, 1)
                negative_pooled_output = self.negative_pooled_output.repeat(self.batch_size, 1)
                
                with self.network.set_temporary_multiplier(0.0):
                    uncond = self.diffusion(noisy_latents, timesteps, negative_encoder_hidden_states, negative_pooled_output, size_condition)
                    cond = self.diffusion(noisy_latents, timesteps, encoder_hidden_states, pooled_output, size_condition)
                    cfg_model_output = uncond + self.config.additional_conf.lcm.guidance_scale * (cond - uncond)

                prev_noisy_latents = self.scheduler.step(noisy_latents, cfg_model_output, timesteps, prev_timesteps, use_ddim=True)

                # target
                target_model_output = self.diffusion(prev_noisy_latents, prev_timesteps, encoder_hidden_states, pooled_output, size_condition)
                if self.tcd:
                    target = self.scheduler.step(prev_noisy_latents, target_model_output, prev_timesteps, inner_timesteps, use_ddim=True)
                else:
                    target = self.scheduler.pred_original_sample(prev_noisy_latents, target_model_output, prev_timesteps)

        loss = nn.functional.mse_loss(pred.float(), target.float(), reduction="mean")

        return loss

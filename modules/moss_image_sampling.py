import torch
from tqdm import tqdm


def denoising_loop(trainer, latents, timesteps, text_output, guidance_scale, controlnet_hint, all_samples):
    """Official CFG: fp32 combination/Euler updates, negative-branch layer 22 skip."""
    images = []
    batch = latents.shape[0]
    latents = latents.float()
    for i, t in enumerate(tqdm(timesteps, desc="Sampling", leave=False)):
        with torch.autocast("cuda", dtype=trainer.autocast_dtype):
            if guidance_scale != 1.0:
                # BaseTrainer assembles negative prompts before positive prompts.
                uncond = trainer.diffusion(latents, t, text_output[:batch], sample=True,
                                           controlnet_hint=controlnet_hint, skip_layers=(22,)).float()
                cond = trainer.diffusion(latents, t, text_output[batch:], sample=True,
                                         controlnet_hint=controlnet_hint).float()
                prediction = uncond + guidance_scale * (cond - uncond)
            else:
                prediction = trainer.diffusion(latents, t, text_output, sample=True,
                                               controlnet_hint=controlnet_hint).float()
        # FlowScheduler.step is the same Euler integration, including the last step to zero.
        prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else t.new_zeros(())
        clean = trainer.scheduler.pred_original_sample(latents, prediction, t)
        latents = trainer.scheduler.step(latents, prediction, t, prev_t)
        if all_samples and i + 1 < len(timesteps):
            with torch.autocast("cuda", dtype=trainer.vae_dtype):
                images.extend(trainer.decode_latents(trainer.denormalize_latents(clean)))
    with torch.autocast("cuda", dtype=trainer.vae_dtype):
        images.extend(trainer.decode_latents(trainer.denormalize_latents(latents)))
    return images

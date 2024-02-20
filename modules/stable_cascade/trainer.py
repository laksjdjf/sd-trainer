import torch
import torch.nn as nn
from torchvision import transforms
from modules.stable_cascade.diffusion_model import CascadeDiffusionModel
from modules.stable_cascade.text_model import CascadeTextModel
from modules.stable_cascade.scheduler import CosineScheduler
from modules.stable_cascade.utils import load_model
from modules.trainer import BaseTrainer

from networks.manager import NetworkManager
from tqdm import tqdm
from diffusers.optimization import get_scheduler
from PIL import Image
import os
import time
import json
import math
import logging
from diffusers import __version__ as diffusers_version

logger = logging.getLogger("トレーナーちゃん")

# 保存先のディレクトリ
DIRECTORIES = [
    "trained",
    "trained/models",
    "trained/networks",
    "trained/controlnet",
]

for directory in DIRECTORIES:
    os.makedirs(directory, exist_ok=True)

class CascadeTrainer(BaseTrainer):
    def __init__(self, config, diffusion:CascadeDiffusionModel, text_model:CascadeTextModel, effnet, scheduler, previewer, network:NetworkManager=None):
        self.config = config
        self.diffusion = diffusion
        self.text_model = text_model
        self.effnet = effnet
        self.previewer = previewer
        self.network = network
        self.diffusers_scheduler = scheduler # モデルのセーブ次にのみ利用
        self.scheduler = CosineScheduler()

    @torch.no_grad()
    def decode_latents(self, latents):
        images_tensor = self.previewer(latents)
        images_numpy = images_tensor.clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
        images = (images_numpy * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    @torch.no_grad()
    def encode_latents(self, images):
        to_tensor_norm = transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )
        images = torch.stack([to_tensor_norm(image) for image in images]).to(self.effnet.device)
        latents = self.effnet(images)
        return latents
    
    def to(self, device="cuda", dtype=torch.float16):
        self.device = device
        self.te_device = device
        
        self.autocast_dtype = dtype
        self.diffusion.unet.to(device, dtype=dtype).eval()
        self.text_model.to(device, dtype=dtype).eval()
        self.effnet.to(device, dtype=dtype).eval()
        self.previewer.to(device, dtype=dtype).eval()

        if self.network:
            self.network.to(device, dtype=dtype).eval()
    
    @torch.no_grad()
    def sample(
        self, 
        prompt="", 
        negative_prompt="", 
        batch_size=1, 
        height=768, 
        width=768, 
        num_inference_steps=30, 
        guidance_scale=7.0, 
        denoise=1.0, 
        seed=4545, 
        images=None, 
        **kwargs
    ):
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state()
        
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        if guidance_scale != 1.0:
            prompt = [negative_prompt] * batch_size + [prompt] * batch_size
        else:
            prompt = [prompt] * batch_size

        timesteps = self.scheduler.set_timesteps(num_inference_steps, self.device)
        timesteps = timesteps[int(num_inference_steps*(1-denoise)):]

        if images is None:
            latents = torch.zeros(batch_size, 16, math.ceil(height / 42.67), math.ceil(width / 42.67), device=self.device, dtype=self.autocast_dtype)
        else:
            with torch.autocast("cuda", dtype=self.autocast_dtype):
                latents = self.encode_latents(images)
            latents.to(dtype=self.autocast_dtype)

        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, timesteps[0])

        self.text_model.to("cuda")
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            encoder_hidden_states, pooled_output = self.text_model(prompt)
        self.text_model.to(self.te_device)

        progress_bar = tqdm(timesteps, desc="Sampling", leave=False, total=len(timesteps))

        for i, t in enumerate(timesteps):
            with torch.autocast("cuda", dtype=self.autocast_dtype):
                latents_input = torch.cat([latents] * (2 if guidance_scale != 1.0 else 1), dim=0)
                model_output = self.diffusion(latents_input, t, encoder_hidden_states, pooled_output)

            if guidance_scale != 1.0:
                uncond, cond = model_output.chunk(2)
                cfg = uncond + guidance_scale * (cond - uncond)
                
                cfg_std, cond_std = cfg.std(), cond.std() 
                cfg_rho = 0.7
                model_output = cfg_rho * (cfg * cond_std / (cfg_std + 1e-9)) + cfg * (1 - cfg_rho)
            
            if i+1 < len(timesteps):
                latents = self.scheduler.step(latents.float(), model_output.float(), t, timesteps[i+1])
            else:
                latents = self.scheduler.pred_original_sample(latents.float(), model_output.float(), t)
            progress_bar.update(1)

        with torch.autocast("cuda", dtype=self.autocast_dtype):
            images = self.decode_latents(latents)
        
        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)
        
        return images
    
    @classmethod
    def from_pretrained(cls, path, effnet_path, previwer_path, config=None, network=None):
        text_model, effnet, unet, scheduler, previewer = load_model(path, effnet_path, previwer_path)
        diffusion = CascadeDiffusionModel(unet)
        return cls(config, diffusion, text_model, effnet, scheduler, previewer, network)


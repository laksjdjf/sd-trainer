import torch
import torch.nn as nn
from torchvision import transforms
from modules.sd3.sd3_diffusion_model import SD3DiffusionModel
from modules.sd3.sd3_text_model import SD3TextModel
from modules.sd3.sd3_scheduler import SD3Scheduler
from modules.utils import get_attr_from_config
from networks.manager import NetworkManager
from tqdm import tqdm
from diffusers import AutoencoderKL, SD3Transformer2DModel
from diffusers.optimization import get_scheduler
from PIL import Image
import os
import time
import json
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

def load_model(path, sdxl=False, clip_skip=-2, revision=None, torch_dtype=None):
    text_model = SD3TextModel.from_pretrained(path, revision=revision, torch_dtype=torch_dtype)
    unet = SD3Transformer2DModel.from_pretrained(path, subfolder='transformer', revision=revision, torch_dtype=torch_dtype)
    vae = AutoencoderKL.from_pretrained(path, subfolder='vae', revision=revision, torch_dtype=torch_dtype)
    scheduler = None
            
    text_model.clip_skip = clip_skip
    return text_model, vae, unet, scheduler

for directory in DIRECTORIES:
    os.makedirs(directory, exist_ok=True)

class SD3Trainer:
    def __init__(self, config, diffusion:SD3DiffusionModel, text_model:SD3TextModel, vae:AutoencoderKL, scheduler, network:NetworkManager):
        self.config = config
        self.diffusion = diffusion
        self.text_model = text_model
        self.vae = vae
        self.network = network
        self.diffusers_scheduler = scheduler # モデルのセーブ次にのみ利用
        self.scheduler = SD3Scheduler()
        self.sdxl = text_model.sdxl
        self.scaling_factor = 1.5305
        self.shift_factor = 0.0609

    @torch.no_grad()
    def decode_latents(self, latents):
        self.vae.to("cuda")
        images = []

        for i in range(latents.shape[0]):
            image = self.vae.decode(latents[i].unsqueeze(0)).sample
            images.append(image)
        images = torch.cat(images, dim=0)
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        self.vae.to(self.vae.device)
        return pil_images

    @torch.no_grad()
    def encode_latents(self, images):
        self.vae.to("cuda")
        to_tensor_norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        images = torch.stack([to_tensor_norm(image) for image in images]).to(self.vae.device)
        latents = self.vae.encode(images).latent_dist.sample()
        self.vae.to(self.vae.device)
        return latents
    
    def to(self, device="cuda", dtype=torch.float16, vae_dtype=None):
        self.device = device
        self.te_device = device
        self.vae_device = device
        
        self.autocast_dtype = dtype
        self.vae_dtype = vae_dtype or dtype

        self.diffusion.unet.to(device, dtype=dtype).eval()
        self.text_model.to(device, dtype=dtype).eval()
        self.vae.to(device, dtype=self.vae_dtype).eval()

        if self.network:
            self.network.to(device, dtype=dtype).eval()
    
    def prepare_lr_scheduler(self, total_steps):
        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=int(0.05 * total_steps),
            num_training_steps=total_steps
        )
        logger.info(f"学習率スケジューラーは{self.lr_scheduler}にした！")
    
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
        controlnet_hint=None, 
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
            latents = torch.zeros(batch_size, 16, height // 8, width // 8, device=self.device, dtype=self.autocast_dtype)
        else:
            with torch.autocast("cuda", dtype=self.vae_dtype):
                latents = self.encode_latents(images) * self.scaling_factor
            latents.to(dtype=self.autocast_dtype)

        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, timesteps[0])

        self.text_model.to("cuda")
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            encoder_hidden_states, pooled_output = self.text_model(prompt)
        self.text_model.to(self.te_device)

        if controlnet_hint is not None:
            if isinstance(controlnet_hint, str):
                controlnet_hint = Image.open(controlnet_hint).convert("RGB")
                controlnet_hint = transforms.ToTensor()(controlnet_hint).unsqueeze(0)
            controlnet_hint = controlnet_hint.to(self.device)
            if guidance_scale != 1.0:
                controlnet_hint = torch.cat([controlnet_hint] *2)
            
            if hasattr(self.network, "set_controlnet_hint"):
                self.network.set_controlnet_hint(controlnet_hint)

        progress_bar = tqdm(timesteps, desc="Sampling", leave=False, total=len(timesteps))

        for i, t in enumerate(timesteps):
            with torch.autocast("cuda", dtype=self.autocast_dtype):
                latents_input = torch.cat([latents] * (2 if guidance_scale != 1.0 else 1), dim=0)
                model_output = self.diffusion(latents_input, t, encoder_hidden_states, pooled_output, controlnet_hint=controlnet_hint)

            if guidance_scale != 1.0:
                uncond, cond = model_output.chunk(2)
                model_output = uncond + guidance_scale * (cond - uncond)
            
            if i+1 < len(timesteps):
                latents = self.scheduler.step(latents, model_output, t, timesteps[i+1])
            else:
                latents = self.scheduler.pred_original_sample(latents, model_output, t)
            progress_bar.update(1)

        with torch.autocast("cuda", dtype=self.vae_dtype):
            images = self.decode_latents(latents / self.scaling_factor + self.shift_factor)
        
        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)
        
        return images
    
    @classmethod
    def from_pretrained(cls, path, sdxl=False, clip_skip=None, config=None, network=None, revision=None, torch_dtype=None):
        if clip_skip is None:
            clip_skip = -2 if sdxl else -1
        text_model, vae, unet, scheduler = load_model(path, sdxl=False, clip_skip=-2, revision=revision, torch_dtype=torch_dtype)
        diffusion = SD3DiffusionModel(unet)
        return cls(config, diffusion, text_model, vae, scheduler, network)


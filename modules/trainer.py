import torch
import torch.nn as nn
from torchvision import transforms
from modules.text_model import BaseTextOutput
from modules.utils import get_attr_from_config, load_model
from networks.manager import NetworkManager
from tqdm import tqdm
from diffusers import AutoencoderKL
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

for directory in DIRECTORIES:
    os.makedirs(directory, exist_ok=True)

class BaseTrainer:
    def __init__(self, config, model_type, diffusion, text_model, vae:AutoencoderKL, diffusers_scheduler, scheduler, network:NetworkManager, nf4=False, taesd=False):
        self.config = config
        self.diffusion = diffusion
        self.text_model = text_model
        self.vae = vae
        self.network = network
        self.diffusers_scheduler = diffusers_scheduler
        self.scheduler = scheduler
        self.model_type = model_type
        self.nf4 = nf4
        self.taesd = taesd

        if model_type == "sd1":
            self.scaling_factor =  0.18215
            self.shift_factor = 0
            self.input_channels = 4
        elif model_type in ["sdxl", "auraflow"]:
            self.scaling_factor = 0.13025
            self.shift_factor = 0
            self.input_channels = 4
        elif model_type == "sd3":
            self.scaling_factor = 1.5305
            self.shift_factor = 0.0609
            self.input_channels = 16
        elif model_type == "flux":
            self.scaling_factor = 0.3611
            self.shift_factor = 0.1159
            self.input_channels = 16

        if config is not None and config.merging_loras:
            for lora in config.merging_loras:
                NetworkManager(
                    text_model=self.text_model,
                    unet=self.diffusion.unet,
                    file_name=lora,
                    mode="merge"
                )

        if config is not None and config.step_range is not None:
            self.min_t, self.max_t = map(float, config.step_range.split(","))
        else:
            self.min_t, self.max_t = 0.0, None
            

    @torch.no_grad()
    def decode_latents(self, latents):
        self.vae.to("cuda")
        images = []
        
        for i in range(latents.shape[0]):
            latent = latents[i].unsqueeze(0).to(self.vae_dtype)
            image = self.vae.decode(latent).sample
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
        images = torch.stack([to_tensor_norm(image) for image in images]).to(self.vae.device, dtype=self.vae_dtype)
        if not self.taesd:
            latents = self.vae.encode(images).latent_dist.sample()
        else:
            latents = self.vae.encode(images).latents
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
    
    def prepare_modules_for_training(self, device="cuda"):
        config = self.config
        self.device = device
        self.te_device = config.te_device or device
        self.vae_device = config.vae_device or device

        self.train_dtype = get_attr_from_config(config.train_dtype)
        self.weight_dtype = get_attr_from_config(config.weight_dtype)
        self.autocast_dtype = get_attr_from_config(config.autocast_dtype) or self.weight_dtype
        self.vae_dtype = get_attr_from_config(config.vae_dtype) or self.weight_dtype
        self.te_dtype = self.train_dtype if config.train_text_encoder else self.weight_dtype
        self.unet_dtype = self.train_dtype if config.train_unet else self.weight_dtype

        logger.info(f"学習対象モデルの型は{self.train_dtype}だって。")
        logger.info(f"学習対象以外の型は{self.weight_dtype}だよ！")
        logger.info(f"オートキャストの型は{self.autocast_dtype}にしちゃった。")

        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.autocast_dtype == torch.float16)

        
        self.text_model.to("cuda", dtype=self.te_dtype)
        if  hasattr(torch, 'float8_e4m3fn') and self.te_dtype== torch.float8_e4m3fn:
            self.text_model.prepare_fp8(self.autocast_dtype) # fp8時のエラー回避

        with torch.autocast("cuda", dtype=self.autocast_dtype), torch.no_grad():
            self.text_model.cache_uncond()
            self.text_model.cache_sample(config.validation_args.prompt, config.validation_args.negative_prompt)
        
        self.text_model.to(self.te_device)
        torch.cuda.empty_cache()
        
        if not self.nf4:
            self.diffusion.unet.to(device, dtype=self.unet_dtype)
        if hasattr(torch, 'float8_e4m3fn') and self.unet_dtype== torch.float8_e4m3fn:
            self.diffusion.prepare_fp8(self.autocast_dtype)

        self.vae.to(self.vae_device, dtype=self.vae_dtype)

        self.diffusion.unet.train(config.train_unet)
        self.diffusion.unet.requires_grad_(config.train_unet)
        self.text_model.train(config.train_text_encoder)
        self.text_model.requires_grad_(config.train_text_encoder)
        self.vae.eval()

    def prepare_network(self, config):
        if config is None:
            self.network = None
            self.network_train = False
            logger.info("ネットワークはないみたい。")
            return 
        
        manager_cls = get_attr_from_config(config.module)
        self.network = manager_cls(
            text_model=self.text_model,
            unet=self.diffusion.unet,
            **config.args
        )

        if config.resume:
            self.network.load_weights(config.resume)
        
        self.network_train = config.train

        self.network.to(self.device, self.train_dtype if self.network_train else self.weight_dtype)
        self.network.train(self.network_train)
        self.network.requires_grad_(self.network_train)

        logger.info("ネットワークを作ったよ！")
    
    def prepare_controlnet(self, config):
        if config is None:
            self.controlnet = None
            self.controlnet_train = False
            logger.info("コントロールネットはないみたい。")
            return
        
        self.diffusion.create_controlnet(config)
        self.controlnet_train = config.train

        self.diffusion.controlnet.to(self.device, self.train_dtype if self.controlnet_train else self.weight_dtype)
        self.diffusion.controlnet.train(self.controlnet_train)
        self.diffusion.controlnet.requires_grad_(self.controlnet_train)

        logger.info("コントロールネットを作ったよ！")

    def apply_module_settings(self):
        if self.config.gradient_checkpointing:
            self.diffusion.enable_gradient_checkpointing()
            self.text_model.enable_gradient_checkpointing()
            self.diffusion.train() # trainでないと適用されない。
            self.text_model.train()
            logger.info("勾配チェックポイントを有効にしてみたよ！")

    def prepare_network_from_file(self, file_name, mode="apply"):
        self.network = NetworkManager(
            text_model=self.text_model,
            unet=self.diffusion.unet,
            file_name=file_name,
            mode=mode
        )

    def prepare_optimizer(self):
        lrs = [float(lr) for lr in self.config.lr.split(",")]
        unet_lr, text_lr = lrs[0], lrs[-1]
        logger.info(f"UNetの学習率は{unet_lr}、text_encoderの学習率は{text_lr}にしてみた！")

        params = []

        if self.config.train_unet:
            params += [{"params":self.diffusion.unet.parameters(), "lr":unet_lr}]
        if self.config.train_text_encoder:
            params += [{"params":self.text_model.parameters(), "lr":text_lr}]
        if self.network_train:
            params += self.network.prepare_optimizer_params(text_lr, unet_lr)
        if self.controlnet_train:
            params += [{"params":self.diffusion.controlnet.parameters(), "lr":unet_lr}]

        optimizer_cls = get_attr_from_config(self.config.optimizer.module)
        self.optimizer = optimizer_cls(params, **self.config.optimizer.args or {})

        logger.info(f"オプティマイザーは{self.optimizer}にしてみた！")
        total_params = sum(p.numel() for group in self.optimizer.param_groups for p in group['params'] if p.requires_grad)
        logger.info(f"学習対象のパラメーター数は{total_params:,}だよ！")

        return params
    
    def prepare_lr_scheduler(self, total_steps):
        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=int(0.05 * total_steps),
            num_training_steps=total_steps
        )
        logger.info(f"学習率スケジューラーは{self.lr_scheduler}にした！")
    
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

        if "controlnet_hint" in batch:
            controlnet_hint = batch["controlnet_hint"].to(self.device)
            if hasattr(self.network, "set_controlnet_hint"):
                self.network.set_controlnet_hint(controlnet_hint)
        else:
            controlnet_hint = None

        if "mask" in batch:
            mask = batch["mask"].to(self.device, dtype = latents.dtype).repeat(1, latents.shape[1], 1, 1)
        else:
            mask = None

        timesteps = self.scheduler.sample_timesteps(latents.shape[0], self.device, self.min_t, self.max_t)
        noise = torch.randn_like(latents)
        if self.config.noise_offset != 0:
            noise += self.config.noise_offset * torch.randn(noise.shape[0], noise.shape[1], 1, 1).to(noise)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            model_output = self.diffusion(noisy_latents, timesteps, text_output, sample=False, size_condition=size_condition, controlnet_hint=controlnet_hint)

        target = self.scheduler.get_target(latents, noise, timesteps) # v_predictionの場合はvelocityになる

        if mask is not None:
            model_output = model_output * mask
            target = target * mask

        loss = nn.functional.mse_loss(model_output.float(), target.float(), reduction="mean")

        return loss
    
    def step(self, batch):
        b_start = time.perf_counter()

        self.optimizer.zero_grad()
        loss = self.loss(batch)
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.lr_scheduler.step()

        if hasattr(self, "loss_ema"):
            self.loss_ema = self.loss_ema * 0.99 + loss.item() * 0.01
        else:
            self.loss_ema = loss.item()

        b_end = time.perf_counter()
        samples_per_second = self.batch_size / (b_end - b_start)

        logs = {"loss_ema":self.loss_ema, "samples_per_second":samples_per_second, "lr":self.lr_scheduler.get_last_lr()[0]}

        return logs
    
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
        all_samples=False,
        **kwargs
    ):
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state()
        output_images = []
        
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        if prompt.split(".")[-1] == "txt":
            with open(prompt, "r") as f:
                prompt = f.read()
                
        if negative_prompt.split(".")[-1] == "txt":
            with open(negative_prompt, "r") as f:
                negative_prompt = f.read()

        if prompt == self.text_model.prompt and negative_prompt == self.text_model.negative_prompt:
            use_cache = True
            positive_output = self.text_model.positive_output.repeat(batch_size)
            if guidance_scale != 1.0:
                negative_output = self.text_model.negative_output.repeat(batch_size)
                text_output = BaseTextOutput.cat([negative_output, positive_output])
            else:
                text_output = positive_output
            text_output = text_output.to(self.device)
        else:
            use_cache = False

        if guidance_scale != 1.0:
            prompt = [negative_prompt] * batch_size + [prompt] * batch_size
        else:
            prompt = [prompt] * batch_size

        timesteps = self.scheduler.set_timesteps(num_inference_steps, self.device)
        timesteps = timesteps[int(num_inference_steps*(1-denoise)):]

        if images is None:
            latents = torch.ones(batch_size, self.input_channels, height // 8, width // 8, device=self.device, dtype=self.autocast_dtype)
        else:
            with torch.autocast("cuda", dtype=self.vae_dtype):
                latents = self.encode_latents(images)
            latents.to(dtype=self.autocast_dtype)
        
        latents = (latents - self.shift_factor) * self.scaling_factor

        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, timesteps[0])
        if not use_cache:
            self.text_model.to("cuda")
            with torch.autocast("cuda", dtype=self.autocast_dtype):
                text_output = self.text_model(prompt)
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
                model_output = self.diffusion(latents_input, t, text_output, sample=True, controlnet_hint=controlnet_hint)

            if guidance_scale != 1.0:
                uncond, cond = model_output.chunk(2)
                model_output = uncond + guidance_scale * (cond - uncond)
            
            if i+1 < len(timesteps):
                latents = self.scheduler.step(latents, model_output, t, timesteps[i+1])
                if all_samples:
                    latents_prev = self.scheduler.pred_original_sample(latents, model_output, t)
                    if not self.taesd:
                        latents_prev = latents_prev / self.scaling_factor + self.shift_factor
                    output_images.extend(self.decode_latents(latents_prev))
            else:
                latents = self.scheduler.pred_original_sample(latents, model_output, t)
            progress_bar.update(1)

        with torch.autocast("cuda", dtype=self.vae_dtype):
            if not self.taesd:
                latents = latents / self.scaling_factor + self.shift_factor
            output_images.extend(self.decode_latents(latents))
        
        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)
        
        return output_images
    
    def save_model(self, output_path):
        logger.info(f"モデルを保存します！")
        if self.config.train_unet or self.config.train_text_encoder:
            self.save_pretrained(os.path.join("trained/models", output_path))
        if self.network_train:
            self.network.save_weights(os.path.join("trained/networks", output_path))
        if self.controlnet_train:
            self.diffusion.controlnet.save_pretrained(os.path.join("trained/controlnet", output_path))

    def sample_validation(self, step):
        logger.info(f"サンプルを生成するよ！")
        images = []
        torch.cuda.empty_cache()
        for i in range(self.config.validation_num_samples):
            image = self.sample(seed=self.config.validation_seed + i, **self.config.validation_args)[0]
            images.append(image)
        torch.cuda.empty_cache()
        return images

    def save_pretrained(self, save_directory):
        if self.model_type not in ["sd1", "sdxl"]:
            raise NotImplementedError(f"{self.model_type}は未対応だよ！")
        self.diffusion.unet.save_pretrained(os.path.join(save_directory, "unet"))
        self.vae.save_pretrained(os.path.join(save_directory, "vae"))
        self.text_model.save_pretrained(save_directory)
        self.diffusers_scheduler.save_pretrained(os.path.join(save_directory, "scheduler"))

        model_index = {}
        model_index["unet"] = ["diffusers", "UNet2DConditionModel"]
        model_index["vae"] = ["diffusers", "AutoencoderKL"]
        model_index["text_encoder"] = ["transformers", "CLIPTextModel"]
        model_index["tokenizer"] = ["transformers", "CLIPTokenizer"]
        model_index["scheduler"] = ["diffusers", "DDIMScheduler"]
        if self.model_type == "sdxl":
            model_index["text_encoder_2"] = ["transformers", "CLIPTextModelWithProjection"]
            model_index["tokenizer_2"] = ["transformers", "CLIPTokenizer"]
            model_index["_class_name"] = "StableDiffusionXLPipeline"
            model_index["force_zeros_for_empty_prompt"] = True
        else:
            model_index["_class_name"] = "StableDiffusionPipeline"

        model_index["_diffusers_version"] = diffusers_version

        with open(os.path.join(save_directory, "model_index.json"), "w") as f:
            json.dump(model_index, f)
    
    @classmethod
    def from_pretrained(cls, path, model_type, clip_skip=None, config=None, network=None, revision=None, torch_dtype=None, variant=None, nf4=False, taesd=False):
        if clip_skip is None:
            clip_skip = -2 if model_type == "sdxl" else -1
        text_model, vae, diffusion, diffusers_scheduler, scheduler = load_model(path, model_type, clip_skip, revision, torch_dtype, variant, nf4, taesd)
        return cls(config, model_type, diffusion, text_model, vae, diffusers_scheduler, scheduler, network, nf4, taesd)

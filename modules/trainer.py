import torch
import torch.nn as nn
from torchvision import transforms
from modules.text import BaseTextOutput
from modules.utils import get_attr_from_config, load_model
from modules.model_registry import get_model_spec
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

# 保存先のディレクトリ(save_model時に作成する)
DIRECTORIES = [
    "trained",
    "trained/models",
    "trained/networks",
    "trained/controlnet",
]

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

        spec = get_model_spec(model_type)
        self.scaling_factor, self.shift_factor = spec.latent_stats(vae)
        self.input_channels = spec.input_channels

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

    def normalize_latents(self, latents):
        # VAE出力のlatentをモデル入力のスケールに変換する
        return (latents - self.shift_factor) * self.scaling_factor

    def denormalize_latents(self, latents):
        # モデル出力のlatentをVAE入力のスケールに戻す(taesdはスケール込みのためそのまま)
        if self.taesd:
            return latents
        return latents / self.scaling_factor + self.shift_factor

    def add_video_dim(self, tensor):
        # animaのようなvideoモデルでは画像テンソルに時間次元を付与する
        if self.model_type == "anima" and tensor.dim() == 4:
            return tensor.unsqueeze(2)
        return tensor

    def latents_from_batch(self, batch):
        # キャッシュ済みlatentがあればそれを、なければ画像をVAEでエンコードして正規化して返す
        if "latents" in batch:
            latents = self.add_video_dim(batch["latents"].to(self.device))
        else:
            with torch.autocast("cuda", dtype=self.vae_dtype), torch.no_grad():
                images = self.add_video_dim(batch["images"].to(self.device))
                latents = self.vae.encode(images).latent_dist.sample()
        return self.normalize_latents(latents)

    def text_output_from_batch(self, batch):
        # キャッシュ済みembeddingがあればそれを、なければキャプションをエンコードして返す
        if "encoder_hidden_states" in batch:
            encoder_hidden_states = batch["encoder_hidden_states"].to(self.device)
            pooled_output = batch["pooled_outputs"].to(self.device)
            return BaseTextOutput(encoder_hidden_states, pooled_output)
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            text_output = self.text_model(batch["captions"])
        return text_output.to(self.device)

    def sample_noise(self, latents):
        noise = torch.randn_like(latents)
        if self.config.noise_offset != 0:
            offset_shape = (noise.shape[0], noise.shape[1]) + (1,) * (noise.dim() - 2)
            noise += self.config.noise_offset * torch.randn(offset_shape).to(noise)
        return noise

    @torch.no_grad()
    def decode_latents(self, latents):
        self.vae.to("cuda")
        images = []
        
        for i in range(latents.shape[0]):
            latent = latents[i].unsqueeze(0).to(self.vae_dtype)
            if self.model_type == "anima":
                image = self.vae.decode(latent, return_dict=False)[0]
            elif len(latent.shape) == 4:
                image = self.vae.decode(latent).sample
            else:
                image = self.vae.tiled_decode(latent).sample
            images.append(image)
        images = torch.cat(images, dim=0)
        images = (images / 2 + 0.5).clamp(0, 1)

        self.vae.to(self.vae.device)

        if self.model_type == "anima" and len(images.shape) == 5:
            images = images[:, :, 0]

        if len(images.shape) == 5:
            videos = []
            for video in images:
                video = video.cpu().permute(1, 2, 3, 0).float().numpy()
                video = (video * 255).round().astype("uint8")
                pil_images = [Image.fromarray(image) for image in video]
                videos.append(pil_images)
            return videos
        else:
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()
            images = (images * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
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
        if isinstance(images[0], list):
            images = torch.stack([[to_tensor_norm(image) for image in video] for video in images]).to(self.vae.device, dtype=self.vae_dtype)
            if self.model_type == "anima":
                images = images.permute(0, 2, 1, 3, 4)
        else:
            images = torch.stack([to_tensor_norm(image) for image in images]).to(self.vae.device, dtype=self.vae_dtype)
            if self.model_type == "anima":
                images = images.unsqueeze(2)
        if not self.taesd:
            latents = self.vae.encode(images).latent_dist.sample()
        else:
            latents = self.vae.encode(images).latents
        self.vae.to(self.vae.device)
        return latents
    
    def to(self, device="cuda", dtype=torch.float16, vae_dtype=None, te_device=None):
        self.device = device
        self.te_device = te_device or device
        self.vae_device = device
        
        self.autocast_dtype = dtype
        self.vae_dtype = vae_dtype or dtype

        self.diffusion.unet.to(device)
        self.text_model.to(self.te_device)
        if not self.nf4:
            self.diffusion.unet.to(dtype=dtype).eval()
            self.text_model.to(dtype=dtype).eval()
            
        self.vae.to(device, dtype=self.vae_dtype).eval()

        if self.network:
            self.network.to(device, dtype=dtype).eval()

        if isinstance(self.scaling_factor, torch.Tensor):
            self.scaling_factor = self.scaling_factor.to(device)
            self.shift_factor = self.shift_factor.to(device)
    
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

        
        if not self.nf4:
            self.text_model.to("cuda" if self.te_device == "cpu" else self.te_device, dtype=self.te_dtype)
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

        if isinstance(self.scaling_factor, torch.Tensor):
            self.scaling_factor = self.scaling_factor.to(device)
            self.shift_factor = self.shift_factor.to(device)

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
        latents = self.latents_from_batch(batch)
        self.batch_size = latents.shape[0] # stepメソッドでも使う
        text_output = self.text_output_from_batch(batch)

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
        noise = self.sample_noise(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        with torch.autocast("cuda", dtype=self.autocast_dtype):
            model_output = self.diffusion(noisy_latents, timesteps, text_output, sample=False, size_condition=size_condition, controlnet_hint=controlnet_hint)

        target = self.scheduler.get_target(latents, noise, timesteps) # v_predictionの場合はvelocityになる

        if mask is not None:
            mask = self.add_video_dim(mask)
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
    
    def _resolve_prompt(self, prompt):
        # .txtファイルが指定された場合は中身をプロンプトとして読み込む
        if prompt.split(".")[-1] == "txt":
            with open(prompt, "r") as f:
                return f.read()
        return prompt

    def _cached_text_output(self, prompt, negative_prompt, batch_size, guidance_scale):
        # 検証用にキャッシュ済みのプロンプトならそのtext_outputを返す。なければNone
        if prompt != self.text_model.prompt or negative_prompt != self.text_model.negative_prompt:
            return None
        positive_output = self.text_model.positive_output.repeat(batch_size)
        if guidance_scale != 1.0:
            negative_output = self.text_model.negative_output.repeat(batch_size)
            text_output = BaseTextOutput.cat([negative_output, positive_output])
        else:
            text_output = positive_output
        return text_output.to(self.device)

    def _encode_sample_prompts(self, prompt, negative_prompt, batch_size, guidance_scale):
        if guidance_scale != 1.0:
            prompts = [negative_prompt] * batch_size + [prompt] * batch_size
        else:
            prompts = [prompt] * batch_size

        if self.te_device == "cpu":
            self.text_model.to("cuda")
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            text_output = self.text_model(prompts)
        text_output = text_output.to(self.device)
        self.text_model.to(self.te_device)
        return text_output

    def _init_sample_latents(self, batch_size, height, width, frames, images):
        if images is None:
            if self.model_type == "anima":
                shape = (batch_size, self.input_channels, 1, height // 8, width // 8)
            elif frames is None:
                shape = (batch_size, self.input_channels, height // 8, width // 8)
            else:
                shape = (batch_size, self.input_channels, (frames + 3) // 4, height // 8, width // 8)
            latents = torch.ones(shape, device=self.device, dtype=self.autocast_dtype)
        else:
            with torch.autocast("cuda", dtype=self.vae_dtype):
                latents = self.encode_latents(images)
        return self.normalize_latents(latents)

    def _prepare_sample_controlnet_hint(self, controlnet_hint, guidance_scale):
        if controlnet_hint is None:
            return None
        if isinstance(controlnet_hint, str):
            controlnet_hint = Image.open(controlnet_hint).convert("RGB")
            controlnet_hint = transforms.ToTensor()(controlnet_hint).unsqueeze(0)
        controlnet_hint = controlnet_hint.to(self.device)
        if guidance_scale != 1.0:
            controlnet_hint = torch.cat([controlnet_hint] * 2)

        if hasattr(self.network, "set_controlnet_hint"):
            self.network.set_controlnet_hint(controlnet_hint)
        return controlnet_hint

    def _denoising_loop(self, latents, timesteps, text_output, guidance_scale, controlnet_hint, all_samples):
        output_images = []
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
                    output_images.extend(self.decode_latents(self.denormalize_latents(latents_prev)))
            else:
                latents = self.scheduler.pred_original_sample(latents, model_output, t)
            progress_bar.update(1)

        with torch.autocast("cuda", dtype=self.vae_dtype):
            output_images.extend(self.decode_latents(self.denormalize_latents(latents)))

        return output_images

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
        frames=None,
        controlnet_hint=None,
        all_samples=False,
        **kwargs
    ):
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state()

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        prompt = self._resolve_prompt(prompt)
        negative_prompt = self._resolve_prompt(negative_prompt)
        text_output = self._cached_text_output(prompt, negative_prompt, batch_size, guidance_scale)

        timesteps = self.scheduler.set_timesteps(num_inference_steps, self.device)
        timesteps = timesteps[int(num_inference_steps*(1-denoise)):]

        latents = self._init_sample_latents(batch_size, height, width, frames, images)
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, timesteps[0])

        if text_output is None:
            text_output = self._encode_sample_prompts(prompt, negative_prompt, batch_size, guidance_scale)

        controlnet_hint = self._prepare_sample_controlnet_hint(controlnet_hint, guidance_scale)

        output_images = self._denoising_loop(latents, timesteps, text_output, guidance_scale, controlnet_hint, all_samples)

        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)

        return output_images
    
    def save_model(self, output_path):
        logger.info(f"モデルを保存します！")
        for directory in DIRECTORIES:
            os.makedirs(directory, exist_ok=True)
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
        if not get_model_spec(self.model_type).supports_save_pretrained:
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
            clip_skip = get_model_spec(model_type).default_clip_skip
        text_model, vae, diffusion, diffusers_scheduler, scheduler = load_model(path, model_type, clip_skip, revision, torch_dtype, variant, nf4, taesd)
        return cls(config, model_type, diffusion, text_model, vae, diffusers_scheduler, scheduler, network, nf4, taesd)

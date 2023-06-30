# python leco.py config.yaml

import sys
import torch
from tqdm import tqdm
import time
from omegaconf import OmegaConf
import importlib
import random
import os

from accelerate.utils import set_seed

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler

from transformers import CLIPTextModel, CLIPTokenizer

# 文字列からモジュールを取得
def get_attr_from_config(config_text: str):
    module = ".".join(config_text.split(".")[:-1])
    attr = config_text.split(".")[-1]
    return getattr(importlib.import_module(module), attr)

# cfgの計算
@torch.no_grad()
def cfg(unet, latents, timesteps, uncond_cond, guidance_scale):
    if guidance_scale == 1:
        return unet(latents, timesteps, uncond_cond.chunk(2)[1]).sample
    elif guidance_scale == 0:
        return unet(latents, timesteps, uncond_cond.chunk(2)[0]).sample
    else:
        noise_pred_uncond, noise_pred_cond = unet(torch.cat([latents]*2), timesteps, uncond_cond).sample.chunk(2)
        return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

def main(config):
    if hasattr(config.train, "seed") and config.train.seed is not None:
        set_seed(config.train.seed)

    lr = config.train.lr

    device = torch.device('cuda')
    weight_dtype = torch.bfloat16 if config.train.amp == 'bfloat16' else torch.float16 if config.train.amp else torch.float32
    print("weight_dtype:", weight_dtype)

    tokenizer = CLIPTokenizer.from_pretrained(config.model.input_path, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(config.model.input_path, subfolder='text_encoder')
    unet = UNet2DConditionModel.from_pretrained(config.model.input_path, subfolder='unet')
    
    if hasattr(config.train, "tome_ratio") and config.train.tome_ratio is not None:
        import tomesd
        tomesd.apply_patch(unet, ratio=config.train.tome_ratio)
       
    if config.train.use_xformers:
        unet.set_use_memory_efficient_attention_xformers(True)
        print("xformersを適用しました。")

    # ampの設定、config.train.ampがFalseなら無効
    scaler = torch.cuda.amp.GradScaler(enabled=not config.train.amp == False)

    params = []  # optimizerに入れるパラメータを格納するリスト

    # networkの準備
    network_class = get_attr_from_config(config.network.module)
    network = network_class(text_encoder, unet, False, False, **config.network.args)
    if config.network.resume is not None:
        network.load_state_dict(torch.load(config.network.resume))
    network.train()
    network.requires_grad_(True)
    params.extend(network.prepare_optimizer_params(lr, lr))
    print(f"{config.network.module}を適用しました。")

    # 最適化関数の設定
    optimizer_class = get_attr_from_config(config.optimizer.module)
    if hasattr(config.optimizer, "args"):
        optimizer = optimizer_class(params, **config.optimizer.args)
    else:
        optimizer = optimizer_class(params)
    print(f"{config.optimizer.module}を適用しました。")

    # モデルの設定
    unet.requires_grad_(False)
    unet.eval()
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    # 勾配チェックポイントによるVRAM削減（計算時間増）
    if config.train.gradient_checkpointing:
        unet.train()
        unet.enable_gradient_checkpointing()
        unet.conv_in.requires_grad_(True)
        print("gradient_checkpointing を適用しました。")

    # 型の指定とGPUへの移動
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    network.to(device, dtype=torch.float32)

    resolution = config.leco.resolution // 8
    batch_size = config.train.batch_size

    noise_scheduler_class = get_attr_from_config(config.leco.noise_scheduler)
    noise_scheduler = noise_scheduler_class.from_pretrained(config.model.input_path, subfolder='scheduler')
    sampling_step = config.leco.sampling_step
    num_samples = config.leco.num_samples
    noise_scheduler.set_timesteps(sampling_step, device=device)

    generate_guidance_scale = config.leco.generate_guidance_scale
    target_guidance_scale = config.leco.target_guidance_scale

    total_steps = config.leco.total_steps
    save_steps = config.leco.save_steps

    lr_scheduler = get_scheduler(
        config.train.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps
    )

    global_steps = 0

    progress_bar = tqdm(range(total_steps), desc="Total Steps", leave=False)
    loss_ema = None  # 訓練ロスの指数平均

    # テキスト埋め込みの生成
    text = [config.leco.target, config.leco.positive if config.leco.positive is not None else config.leco.target, ""]
    tokens = tokenizer(text, max_length=tokenizer.model_max_length, padding="max_length",
                        truncation=True, return_tensors='pt').input_ids.to(device)
    encoder_hidden_states = text_encoder(tokens, output_hidden_states=True).last_hidden_state.to(device)
    target, positive, uncond = encoder_hidden_states.chunk(3)
    target = target.repeat(batch_size, 1, 1)
    positive = positive.repeat(batch_size, 1, 1)
    uncond = uncond.repeat(batch_size, 1, 1)
    uncond_positive = torch.cat([uncond, positive], dim=0)

    latents_and_times = []
    loss_ema = None

    for step in range(total_steps):
        b_start = time.perf_counter()
        # デノイズ途中の潜在変数を生成
        with torch.autocast("cuda", enabled=True): #生成時は強制AMP 
            if len(latents_and_times) == 0:
                # x_T
                latents = torch.randn(batch_size, 4, resolution, resolution, device=device, dtype=weight_dtype)
 
                timestep_to = random.sample(range(sampling_step), num_samples) # tをnum_samples個サンプリング
                timestep_to.sort()
                timedelta = random.choice(range(1000//sampling_step-1)) # 全ステップ学習できるようちょっとずらす
                target_index = 0
                for i, t in tqdm(enumerate(noise_scheduler.timesteps[0:timestep_to[-1]+1])):
                    timestep = t + timedelta
                    latents_input = noise_scheduler.scale_model_input(latents, timestep)
                    noise_pred = cfg(unet, latents_input, timestep, uncond_positive, generate_guidance_scale)
                    latents = noise_scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
                    if i == timestep_to[target_index]:
                        target_index += 1
                        latents_and_times.append((latents, timestep))
                        
        with torch.autocast("cuda", enabled=not config.train.amp == False): 
            latents, timesteps = latents_and_times.pop()
            with network.no_apply():
                noise_pred_positive = cfg(unet, latents, timesteps, uncond_positive, target_guidance_scale)
            noise_pred = unet(latents, timesteps, target).sample
        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise_pred_positive.float(), reduction="mean")
        
        if loss_ema is None:
            loss_ema = loss.item()
        else:
            loss_ema = loss_ema * 0.9 + loss.item() * 0.1 # 指数平均

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()

        global_steps += 1

        b_end = time.perf_counter()
        samples_per_second = batch_size / (b_end - b_start)

        logs = {"loss": loss_ema, "samples_per_second": samples_per_second, "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.update(1)
        progress_bar.set_postfix(logs)

        if global_steps % save_steps == 0:
            os.makedirs(os.path.join("trained","networks"), exist_ok=True)
            network.save_weights(os.path.join("trained","networks",config.model.output_name), torch.float16)

if __name__ == "__main__":
    config = OmegaConf.load(sys.argv[1])
    main(config)

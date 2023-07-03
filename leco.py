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

from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from diffusers.optimization import get_scheduler

from transformers import CLIPTextModel, CLIPTokenizer

from leco_utils.leco_dataset import TextEmbeddingDataset

from utils.model import load_model

# 文字列からモジュールを取得
def get_attr_from_config(config_text: str):
    module = ".".join(config_text.split(".")[:-1])
    attr = config_text.split(".")[-1]
    return getattr(importlib.import_module(module), attr)

# cfgの計算
@torch.no_grad()
def cfg(unet, latents, timesteps, positive_negative, guidance_scale, neutral=None):
    if neutral is not None:
        positive_negative_neutral = torch.cat([positive_negative, neutral], dim=0)
        positive, negative, neutral = unet(torch.cat([latents]*3), timesteps, positive_negative_neutral).sample.chunk(3)
        return neutral + guidance_scale * (positive - negative)
    elif guidance_scale == 1:
        return unet(latents, timesteps, positive_negative.chunk(2)[0]).sample
    elif guidance_scale == 0:
        return unet(latents, timesteps, positive_negative.chunk(2)[1]).sample
    else:
        positive, negative = unet(torch.cat([latents]*2), timesteps, positive_negative).sample.chunk(2)
        return negative + guidance_scale * (positive - negative)
    
def collate_fn(batch):
    return batch[0]

def main(config):
    if hasattr(config.train, "seed") and config.train.seed is not None:
        set_seed(config.train.seed)

    lr = config.train.lr

    device = torch.device('cuda')
    weight_dtype = torch.bfloat16 if config.train.amp == 'bfloat16' else torch.float16 if config.train.amp else torch.float32
    print("weight_dtype:", weight_dtype)

    tokenizer, text_encoder, _, unet, scheduler = load_model(config.model.input_path)
    noise_scheduler_class = get_attr_from_config(config.leco.noise_scheduler)
    noise_scheduler = noise_scheduler_class.from_config(scheduler.config)
    
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

    sampling_step = config.leco.sampling_step
    num_samples = config.leco.num_samples
    noise_scheduler.set_timesteps(sampling_step, device=device)

    generate_guidance_scale = config.leco.generate_guidance_scale

    # テキスト埋め込みの生成
    print("プロンプトを処理します。")
    prompts = OmegaConf.load(config.leco.prompts_file)
    dataset = TextEmbeddingDataset(prompts, tokenizer, text_encoder, device, batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
    
    del text_encoder, tokenizer
    
    total_steps = config.leco.epochs * len(dataloader)
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

    latents_and_times = {i:[] for i in range(len(dataloader))}

    for epoch in range(config.leco.epochs):
        for idx, batch in enumerate(dataloader):
            b_start = time.perf_counter()

            target = batch["target"]
            positive = batch["positive"]
            negative = batch["negative"]
            neutral = batch["neutral"]
            guidance_scale = batch["guidance_scale"]

            # デノイズ途中の潜在変数を生成
            with torch.autocast("cuda", enabled=True): #生成時は強制AMP 
                if len(latents_and_times[idx]) == 0:
                    # x_T
                    latents = torch.randn(batch_size, 4, resolution, resolution, device=device, dtype=weight_dtype)
    
                    timestep_to = random.sample(range(sampling_step), num_samples) # tをnum_samples個サンプリング
                    timestep_to.sort()
                    timedelta = random.choice(range(1000//sampling_step-1)) # 全ステップ学習できるようちょっとずらす
                    target_index = 0
                    for i, t in tqdm(enumerate(noise_scheduler.timesteps[0:timestep_to[-1]+1])):
                        timestep = t + timedelta
                        latents_input = noise_scheduler.scale_model_input(latents, timestep)
                        noise_pred = cfg(unet, latents_input, timestep, torch.cat([target, negative],dim=0), generate_guidance_scale)
                        latents = noise_scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
                        if i == timestep_to[target_index]:
                            target_index += 1
                            latents_and_times[idx].append((latents, timestep))
                            
            with torch.autocast("cuda", enabled=not config.train.amp == False): 
                latents, timesteps = latents_and_times[idx].pop()
                with network.set_temporary_multiplier(0.0):
                    noise_pred_positive = cfg(unet, latents, timesteps, torch.cat([positive, negative],dim=0), guidance_scale, neutral=neutral)
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

            if global_steps % save_steps == 0 and not global_steps == total_steps:
                print("セーブしますぅ")
                os.makedirs(os.path.join("trained","networks"), exist_ok=True)
                network.save_weights(os.path.join("trained","networks",config.model.output_name), torch.float16)
                
    print("セーブしますぅ")
    os.makedirs(os.path.join("trained","networks"), exist_ok=True)
    network.save_weights(os.path.join("trained","networks",config.model.output_name), torch.float16)

if __name__ == "__main__":
    config = OmegaConf.load(sys.argv[1])
    main(config)

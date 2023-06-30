# python main.py config.yaml

import sys
import torch
import os
from tqdm import tqdm
import numpy as np
import time
from omegaconf import OmegaConf
import importlib

from accelerate.utils import set_seed

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler

from transformers import CLIPTextModel, CLIPTokenizer

# データローダー用の関数
def collate_fn(x):
    return x[0]

# 文字列からモジュールを取得
def get_attr_from_config(config_text: str):
    module = ".".join(config_text.split(".")[:-1])
    attr = config_text.split(".")[-1]
    return getattr(importlib.import_module(module), attr)


def main(config):
    if hasattr(config.train, "seed") and config.train.seed is not None:
        set_seed(config.train.seed)

    lrs = config.train.lr.split(",")
    text_lr, unet_lr = float(lrs[0]), float(lrs[-1])  # 長さが1の場合同じ値になる

    device = torch.device('cuda')
    weight_dtype = torch.bfloat16 if config.train.amp == 'bfloat16' else torch.float16 if config.train.amp else torch.float32
    print("weight_dtype:", weight_dtype)

    tokenizer = CLIPTokenizer.from_pretrained(config.model.input_path, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(config.model.input_path, subfolder='text_encoder')
    vae = AutoencoderKL.from_pretrained(config.model.input_path, subfolder='vae')
    vae.enable_slicing()
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
    if hasattr(config, "network"):
        network_class = get_attr_from_config(config.network.module)
        network = network_class(text_encoder, unet, config.feature.up_only, config.train.train_encoder, **config.network.args)
        if config.network.resume is not None:
            network.load_state_dict(torch.load(config.network.resume))
        network.train(config.network.train)
        network.requires_grad_(config.network.train)
        if config.network.train:
            params.extend(network.prepare_optimizer_params(text_lr, unet_lr))
        print(f"{config.network.module}を適用しました。")
    else:
        network = None

    # pfgの準備
    if hasattr(config, "pfg"):
        from networks.pfg import PFGNetwork
        pfg = PFGNetwork(**config.pfg.args)
        if config.pfg.resume is not None:
            pfg.load_state_dict(torch.load(config.pfg.resume))
        pfg.train(config.pfg.train)
        pfg.requires_grad_(config.pfg.train)
        if config.pfg.train:
            params.append({'params': pfg.parameters(), 'lr': unet_lr})
        print("pfgを適用しました。")
    else:
        pfg = None

    # unet, text encoderのパラメータを追加
    if config.train.train_unet:
        if config.feature.up_only:
            params.append({'params': unet.up_blocks.parameters(), 'lr': unet_lr})
        else:
            params.append({'params': unet.parameters(), 'lr': unet_lr})
            if config.train.train_encoder:
                params.append({'params': text_encoder.parameters(), 'lr': text_lr})

    # controlnetの準備
    if hasattr(config, "controlnet"):
        from diffusers import ControlNetModel
        if config.controlnet.resume is not None:
            controlnet = ControlNetModel.from_pretrained(config.controlnet.resume)
        else:
            controlnet = ControlNetModel.from_unet(unet)
        controlnet.config.global_pool_conditions = config.controlnet.global_average_pooling
        if config.train.use_xformers:
            controlnet.enable_xformers_memory_efficient_attention()
        controlnet.train(config.controlnet.train)
        controlnet.requires_grad_(config.controlnet.train)
        if config.controlnet.train:
            params.append({'params': controlnet.parameters(), 'lr': unet_lr})
        print("controlnetを適用しました。")
    else:
        controlnet = None

    # 最適化関数の設定
    optimizer_class = get_attr_from_config(config.optimizer.module)
    if hasattr(config.optimizer, "args"):
        optimizer = optimizer_class(params, **config.optimizer.args)
    else:
        optimizer = optimizer_class(params)
    print(f"{config.optimizer.module}を適用しました。")

    # モデルの設定
    vae.requires_grad_(False)
    vae.eval()
    unet.requires_grad_(config.train.train_unet)
    unet.train(config.train.train_unet)
    text_encoder.requires_grad_(config.train.train_encoder)
    text_encoder.train(config.train.train_encoder)

    if config.feature.up_only and network is not None:
        unet.requires_grad_(False)
        unet.eval()
        unet.up_blocks.requires_grad_(True)
        unet.up_blocks.train()

    # 勾配チェックポイントによるVRAM削減（計算時間増）
    if config.train.gradient_checkpointing:
        unet.train()
        unet.enable_gradient_checkpointing()
        if config.train.train_encoder:
            text_encoder.text_model.embeddings.requires_grad_(True)  # 先頭のモジュールが勾配有効である必要があるらしい
            text_encoder.train() #trainがTrueである必要があるらしい
            text_encoder.gradient_checkpointing_enable()
        else:
            if not config.feature.up_only:
                unet.conv_in.requires_grad_(True)
        if controlnet is not None:
            controlnet.enable_gradient_checkpointing()
        print("gradient_checkpointing を適用しました。")

    # 型の指定とGPUへの移動
    text_encoder.to(device, dtype=torch.float32 if config.train.train_encoder else weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=torch.float32 if config.train.train_unet else weight_dtype)
    if network is not None:
        network.to(device, dtype=torch.float32 if config.network.train else weight_dtype)
    if pfg is not None:
        pfg.to(device, dtype=torch.float32 if config.pfg.train else weight_dtype)
    if controlnet is not None:
        controlnet.to(device, dtype=torch.float32 if config.controlnet.train else weight_dtype)

    noise_scheduler = DDPMScheduler.from_pretrained(config.model.input_path, subfolder='scheduler')

    # sampling stepの範囲を指定
    step_range = [int(float(step)*noise_scheduler.num_train_timesteps) for step in config.feature.step_range.split(",")]

    dataset_class = get_attr_from_config(config.dataset.module)
    dataset = dataset_class(tokenizer, config, **config.dataset.args)

    dataloader_class = get_attr_from_config(config.dataset.loader.module)
    dataloader = dataloader_class(
        dataset, collate_fn=collate_fn if config.dataset.loader.collate_fn == "identity" else None, **config.dataset.loader.args)

    step_per_epoch = len(dataloader) // config.feature.minibatch_repeat
    total_steps = config.train.epochs * step_per_epoch if config.feature.test_steps <= 0 else config.feature.test_steps

    lr_scheduler = get_scheduler(
        config.train.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps
    )

    save_module = get_attr_from_config(config.save.module)
    save = save_module(config, step_per_epoch, **config.save.args)

    global_steps = 0

    progress_bar = tqdm(range(total_steps), desc="Total Steps", leave=False)
    loss_ema = None  # 訓練ロスの指数平均

    # ミニバッチリピートがnだと1回のループでnエポック進む扱い。
    for epoch in range(0, config.train.epochs, config.feature.minibatch_repeat):
        for batch in dataloader:

            b_start = time.perf_counter()

            tokens = tokenizer(batch["captions"], max_length=tokenizer.model_max_length, padding="max_length",
                               truncation=True, return_tensors='pt').input_ids.to(device)
            encoder_hidden_states = text_encoder(tokens, output_hidden_states=True).last_hidden_state.to(device)

            if 'latents' in batch: # 事前に計算した潜在変数を使う場合
                latents = batch['latents'].to(device) * 0.18215
            else:
                latents = vae.encode(batch['image'].to(device, dtype=weight_dtype)).latent_dist.sample().to(device) * 0.18215

            if "pfg" in batch: 
                pfg_inputs = batch["pfg"].to(device)
                with torch.autocast("cuda", enabled=not config.train.amp == False):
                    pfg_feature = pfg(pfg_inputs).to(dtype=encoder_hidden_states.dtype)
                encoder_hidden_states = torch.cat([encoder_hidden_states, pfg_feature], dim=1)

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # step_rangeの範囲内でランダムにstepを選択
            timesteps = torch.randint(step_range[0], step_range[1], (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.autocast("cuda", enabled=not config.train.amp == False):
                if controlnet is not None:
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=batch["control"].to(latents.device),
                        return_dict=False,
                    )
                else:
                    down_block_res_samples, mid_block_res_sample = None, None

                noise_pred = unet(noisy_latents,
                                  timesteps,
                                  encoder_hidden_states,
                                  down_block_additional_residuals=down_block_res_samples,
                                  mid_block_additional_residual=mid_block_res_sample,
                                  ).sample

            if config.model.v_prediction:
                noise = noise_scheduler.get_velocity(latents, noise, timesteps)

            if "mask" in batch:
                mask = batch["mask"].to(device)

                noise = noise * mask
                noise_pred = noise_pred * mask

            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

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
            samples_per_second = bsz / (b_end - b_start)

            logs = {"loss": loss_ema, "samples_per_second": samples_per_second, "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.update(1)
            progress_bar.set_postfix(logs)

            final = total_steps == global_steps # 最後のステップかどうか
            save(global_steps, final, logs, batch, text_encoder, unet, vae, tokenizer, noise_scheduler, network, pfg, controlnet)
            if final:
                return

if __name__ == "__main__":
    config = OmegaConf.load(sys.argv[1])
    main(config)

# python main.py config.yaml

import sys
import torch
from tqdm import tqdm
import time
from omegaconf import OmegaConf

from accelerate.utils import set_seed

from diffusers import  DDPMScheduler
from diffusers.optimization import get_scheduler

from utils.model import load_model, patch_mid_block_checkpointing
from utils.functions import collate_fn, get_attr_from_config, default

def main(config):
    seed = default(config.train, "seed", None)
    if seed is not None:
        set_seed(seed)

    sdxl = default(config.model, "sdxl", False)

    lrs = config.train.lr.split(",")
    text_lr, unet_lr = float(lrs[0]), float(lrs[-1])  # 長さが1の場合同じ値になる
    print("text_lr:", text_lr, "unet_lr:", unet_lr)

    device = torch.device('cuda') # 訓練デバイス
    
    # 学習対象以外の型
    weight_dtype = torch.bfloat16 if config.train.amp == 'bfloat16' else torch.float16 if config.train.amp else torch.float32
    
    full_bf16 = default(config.train, "full_bf16", False)
    if full_bf16:
        assert weight_dtype == torch.bfloat16, "full_bf16を使う場合、amp: bfloat16を設定してください。"
    
    train_dtype = weight_dtype if full_bf16 else torch.float32 # 学習対象や勾配の型
    
    amp = (not config.train.amp == False) and not full_bf16
    
    print("weight_dtype:", weight_dtype)
    print("train_dtype:", train_dtype)

    text_model, vae, unet, scheduler = load_model(config.model.input_path, sdxl)
    text_model.clip_skip = default(config.model, "clip_skip", -1)
    
    vae.enable_slicing()
    latent_scale = 0.13025 if sdxl else 0.18215 # いずれvaeのconfigから取得するようにしたい
    print(f"vae scale factor:{latent_scale}")

    noise_scheduler = DDPMScheduler.from_config(scheduler.config)

    # LoRAの事前マージ
    if default(config.model, "add_lora", False):
        from networks.lora import LoRANetwork
        LoRANetwork.from_file(text_model, unet, config.model.add_lora, mode="merge")

    if default(config.train, "tome_ratio", False):
        import tomesd
        tomesd.apply_patch(unet, ratio=config.train.tome_ratio)
        print(f"tomeを適用しました。ratio={config.train.tome_ratio}")

    if config.train.use_xformers:
        unet.set_use_memory_efficient_attention_xformers(True)
        print("xformersを適用しました。")

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    params = []  # optimizerに入れるパラメータを格納するリスト

    # networkの準備
    if hasattr(config, "network"):
        network_class = get_attr_from_config(config.network.module)
        if config.network.resume is None:
            network = network_class(text_model, unet, config.feature.up_only, config.train.train_encoder, **config.network.args)
        else:
            network = network_class.from_file(text_model, unet, config.network.resume)
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
    
    # ip-adapterの準備
    if hasattr(config, "ip_adapter"):
        from networks.ip_adapter import IPAdapter
        ip_adapter = IPAdapter(unet, **config.ip_adapter.args)
        ip_adapter.train(config.ip_adapter.train)
        ip_adapter.requires_grad_(config.ip_adapter.train)
        if config.ip_adapter.train:
            params.append({'params': ip_adapter.trainable_params(), 'lr': unet_lr})
        print("ip-adapterを適用しました。")
    else:
        ip_adapter = None

    # unet, text encoderのパラメータを追加
    if config.train.train_unet:
        if config.feature.up_only:
            params.append({'params': unet.up_blocks.parameters(), 'lr': unet_lr})
        else:
            params.append({'params': unet.parameters(), 'lr': unet_lr})
            if config.train.train_encoder:
                params.append({'params': text_model.parameters(), 'lr': text_lr})

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

    # textual inversionの準備
    if hasattr(config, "textual_inversion"):
        textual_inversion = True
        text_model.prepare_textual_inversion(**config.textual_inversion.args)
        for param in text_model.get_textual_inversion_params():
            params.append({'params': param, 'lr': text_lr})
    else:
        textual_inversion = False

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
    if textual_inversion:
        text_model.set_textual_inversion_mode()
    else:
        text_model.requires_grad_(config.train.train_encoder and network is None)
        text_model.train(config.train.train_encoder and network is None)

    if config.feature.up_only and network is None:
        unet.requires_grad_(False)
        unet.eval()
        unet.up_blocks.requires_grad_(True)
        unet.up_blocks.train()

    # 勾配チェックポイントによるVRAM削減（計算時間増）
    if config.train.gradient_checkpointing:
        unet.train()
        unet.enable_gradient_checkpointing()
        patch_mid_block_checkpointing(unet.mid_block)
        if config.train.train_encoder:
            text_model.text_encoder.text_model.embeddings.requires_grad_(True)  # 先頭のモジュールが勾配有効である必要があるらしい
            if text_model.text_encoder_2 is not None:
                text_model.text_encoder_2.text_model.embeddings.requires_grad_(True)
            text_model.train() #trainがTrueである必要があるらしい
            text_model.gradient_checkpointing_enable()
        else:
            if not config.feature.up_only:
                unet.conv_in.requires_grad_(True)
        if controlnet is not None:
            controlnet.enable_gradient_checkpointing()
        print("gradient_checkpointing を適用しました。")

    # 型の指定とGPUへの移動
    vae_device = torch.device('cpu') if default(config.train, "vae_offload", False) else device
    text_encoder_device = torch.device('cpu') if default(config.train, "text_encoder_offload", False) else device
    text_model.to(text_encoder_device, dtype=train_dtype if (config.train.train_encoder or textual_inversion) else weight_dtype)
    vae.to(vae_device, dtype=weight_dtype)
    unet.to(device, dtype=train_dtype if config.train.train_unet else weight_dtype)
    if network is not None:
        network.to(device, dtype=train_dtype if config.network.train else weight_dtype)
    if pfg is not None:
        pfg.to(device, dtype=train_dtype if config.pfg.train else weight_dtype)
    if controlnet is not None:
        controlnet.to(device, dtype=train_dtype if config.controlnet.train else weight_dtype)
    if ip_adapter is not None:
        ip_adapter.to(device, dtype=train_dtype if config.ip_adapter.train else weight_dtype)

    # sampling stepの範囲を指定
    step_range = [int(float(step)*noise_scheduler.num_train_timesteps) for step in config.feature.step_range.split(",")]

    dataset_class = get_attr_from_config(config.dataset.module)
    dataset = dataset_class(config, text_model, **config.dataset.args)

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
    save = save_module(config, text_model, step_per_epoch, **config.save.args)

    global_steps = 0

    progress_bar = tqdm(range(total_steps), desc="Total Steps", leave=False)
    loss_ema = None  # 訓練ロスの指数平均
    
    torch.cuda.empty_cache()

    # ミニバッチリピートがnだと1回のループでnエポック進む扱い。
    for _ in range(0, config.train.epochs, config.feature.minibatch_repeat):
        for batch in dataloader:
            b_start = time.perf_counter()

            if 'encoder_hidden_states' in batch:
                encoder_hidden_states = batch["encoder_hidden_states"].to(device, dtype=train_dtype)
                pooled_output = batch["pooled_outputs"].to(device, dtype=train_dtype)
            else:
                encoder_hidden_states, pooled_output = text_model.encode_text(batch["captions"])
            if 'latents' in batch: # 事前に計算した潜在変数を使う場合
                latents = batch['latents'].to(device, dtype=train_dtype) * latent_scale
            else:
                latents = vae.encode(batch['images'].to(device, dtype=weight_dtype)).latent_dist.sample().to(device) * latent_scale
            
            if "pfg" in batch: 
                pfg_inputs = batch["pfg"].to(device, dtype=weight_dtype)
                with torch.autocast("cuda", enabled=amp, dtype=weight_dtype):
                    pfg_feature = pfg(pfg_inputs).to(dtype=encoder_hidden_states.dtype)
                encoder_hidden_states = torch.cat([encoder_hidden_states, pfg_feature], dim=1)

            if "image_embeds" in batch:
                image_embeds = batch["image_embeds"].to(device, dtype=weight_dtype)
                with torch.autocast("cuda", enabled=amp, dtype=weight_dtype):
                    image_embeds = ip_adapter.get_image_embeds(image_embeds)
                    ip_adapter.set_ip_hidden_states(image_embeds)
            noise = torch.randn_like(latents)
            
            if default(config.train,"noise_offset", False):
                noise_offset = config.train.noise_offset * torch.randn(latents.shape[0], latents.shape[1], 1, 1)
                noise = noise + noise_offset.to(noise.device, dtype=noise.dtype)

            bsz = latents.shape[0]

            if sdxl:
                size_condition = batch["size_condition"].to(latents.device, dtype=latents.dtype)
                added_cond_kwargs = {"text_embeds": pooled_output, "time_ids": size_condition}
            else:
                added_cond_kwargs = None
                
            # step_rangeの範囲内でランダムにstepを選択
            timesteps = torch.randint(step_range[0], step_range[1], (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.autocast("cuda", enabled=amp, dtype=weight_dtype):
                if controlnet is not None:
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=batch["control"].to(latents.device, dtype=latents.dtype),
                        return_dict=False,
                    )
                else:
                    down_block_res_samples, mid_block_res_sample = None, None

                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

            if config.model.v_prediction:
                noise = noise_scheduler.get_velocity(latents, noise, timesteps)

            if "mask" in batch:
                mask = batch["mask"].to(device, dtype = latents.dtype)
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

            if textual_inversion:
                text_model.reset_unupdate_embedding()

            global_steps += 1

            b_end = time.perf_counter()
            samples_per_second = bsz / (b_end - b_start)

            logs = {"loss": loss_ema, "samples_per_second": samples_per_second, "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.update(1)
            progress_bar.set_postfix(logs)

            final = total_steps == global_steps # 最後のステップかどうか
            save(global_steps, final, logs, batch, text_model, unet, vae, noise_scheduler, network, pfg, controlnet, ip_adapter)
            if final:
                return

if __name__ == "__main__":
    config = OmegaConf.load(sys.argv[1])
    main(config)
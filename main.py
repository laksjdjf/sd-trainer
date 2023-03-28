import argparse
import torch
import os
from tqdm import tqdm
import numpy as np
import time
from omegaconf import OmegaConf
import importlib

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler

from transformers import CLIPTextModel, CLIPTokenizer

parser = argparse.ArgumentParser(description='訓練コード')
parser.add_argument('config', type=str, help='config パス')

def collate_fn(x):
    return x[0]

#メイン処理
def main(args):
    ###学習準備##############################################################################
    config = OmegaConf.load(args.config)
    
    #ミニバッチサイズ数を計算
    minibatch_size = config.train.batch_size // config.feature.minibatch_repeat
    
    #学習率を定義
    lrs = config.train.lr.split(",")
    text_lr, unet_lr = float(lrs[0]), float(lrs[-1]) #長さが1の場合同じ値になる
    
    #device, dtypeを定義
    device = torch.device('cuda') # cuda以外の場合があるのか？
    weight_dtype = torch.float16 if config.train.amp else torch.float32
    
    #モデルのロード
    tokenizer = CLIPTokenizer.from_pretrained(config.model.input_path, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(config.model.input_path, subfolder='text_encoder')
    vae = AutoencoderKL.from_pretrained(config.model.input_path, subfolder='vae')
    vae.enable_slicing()
    unet = UNet2DConditionModel.from_pretrained(config.model.input_path, subfolder='unet')
    
    if config.train.use_xformers:
        unet.set_use_memory_efficient_attention_xformers(True)
        print("xformersを適用しました。")
    
    #AMP用のスケーラー
    scaler = torch.cuda.amp.GradScaler(enabled=config.train.amp)
    
    #optimizerに入れるパラメータ
    params = []
    #networkの準備、パラメータの確定
    if config.network is not None:
        network_class = getattr(importlib.import_module(config.network.module), config.network.attribute)
        network = network_class(text_encoder, unet, config.feature.up_only, config.train.train_encoder, **config.network.args)
        if config.network.resume is not None:
            network.load_state_dict(torch.load(config.network.resume))
        params.extend(network.prepare_optimizer_params(text_lr,unet_lr))
    else:
        network = None
    
    #pfgの準備
    if config.pfg is not None:
        from pfg.pfg import PFGNetwork
        pfg = PFGNetwork(**config.pfg.args)
        if config.pfg.resume is not None:
            pfg.load_state_dict(torch.load(config.pfg.resume))
        params.append({'params':pfg.parameters(), 'lr':unet_lr})
    else:
        pfg = None
    
    #unetのパラメータを追加
    if config.train.train_unet:
        if config.feature.up_only:
            params.append({'params':unet.up_blocks.parameters(), 'lr':unet_lr})
        else:
            params.append({'params':unet.parameters(), 'lr':unet_lr})
            if config.train.train_encoder:
                params.append({'params':text_encoder.parameters(), 'lr':text_lr})
    
    #最適化関数
    try:
        optimizer_class = getattr(importlib.import_module(config.optimizer.module), config.optimizer.attribute)
        optimizer = optimizer_class(params,**config.optimizer.args)
        print(f"optimizer:{config.optimizer.attribute}を適用しました。")
    except:
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(params)
        print(f"optimizer:{config.optimizer.attribute}を適用できなかったので、通常のAdamWを使用します。")

    
    #勾配、trainとevalの確定
    vae.requires_grad_(False)
    vae.eval()
    unet.requires_grad_(config.train.train_unet)
    unet.train(config.train.train_unet)
    text_encoder.requires_grad_(config.train.train_encoder)
    text_encoder.train(config.train.train_encoder)
    
    if config.feature.up_only and config.network is None:
        unet.requires_grad_(False)
        unet.eval()
        unet.up_blocks.requires_grad_(True)
        unet.up_blocks.train()
    
    #勾配チェックポイントによるVRAM削減（計算時間増）
    if config.train.gradient_checkpointing:
        if config.train.train_encoder:
            text_encoder.text_model.embeddings.requires_grad_(True) #先頭のモジュールが勾配有効である必要があるらしい
            unet.enable_gradient_checkpointing()
            text_encoder.gradient_checkpointing_enable()
            print("gradient_checkpointing を適用しました。")
        else:
            if config.feature.up_only:
                unet.conv_in.requires_grad_(True)
            unet.enable_gradient_checkpointing()
            print("gradient_checkpointing を適用しました。")
    
    #型の指定とGPUへの移動
    text_encoder.to(device,dtype=torch.float32 if config.train.train_encoder else weight_dtype)
    vae.to(device,dtype=weight_dtype)
    unet.to(device,dtype=torch.float32 if config.train.train_unet else weight_dtype)
    if config.network is not None:
        network.to(device,dtype=torch.float32)
    if config.pfg is not None:
        pfg.to(device,dtype=torch.float32)
    
    #ノイズスケジューラー
    noise_scheduler = DDPMScheduler.from_pretrained(
        config.model.input_path,
        subfolder='scheduler',
    )
    
    #sampling stepの範囲を指定
    step_range = [int(float(step)*noise_scheduler.num_train_timesteps) for step in config.feature.step_range.split(",")]
    
    #データローダー
    dataset_class = getattr(importlib.import_module(config.dataset.module), config.dataset.attribute)
    dataset = dataset_class(tokenizer, config.train.batch_size, **config.dataset.args)
    
    dataloader_class = getattr(importlib.import_module(config.dataset.loader.module), config.dataset.loader.attribute)
    dataloader = dataloader_class(dataset, collate_fn = collate_fn if config.dataset.loader.collate_fn == "identity" else None, **config.dataset.loader.args)
    
    #Tトータルステップ
    total_steps = (config.train.epochs // config.feature.minibatch_repeat) * len(dataloader)
    
    #学習率スケジューラー
    lr_scheduler = get_scheduler(
        config.train.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps= int(0.05 * total_steps), #そこまで重要に思えないから0.05固定
        num_training_steps= total_steps
    )
    
    #save
    save_module = getattr(importlib.import_module(config.save.module), config.save.attribute)
    save = save_module(config.model.output_path, config.train.train_unet, len(dataloader) // config.feature.minibatch_repeat, **config.save.args)
    
    #全ステップ
    global_steps = 0
    
    #プログレスバー
    progress_bar = tqdm(range(total_steps), desc="Total Steps", leave=False)
    loss_ema = None #訓練ロスの指数平均
    
    #学習ループ
    for epoch in range(0,config.train.epochs,config.feature.minibatch_repeat): #ミニバッチリピートがnだと1回のループでnエポック進む扱い。
        for batch in dataloader:
            #時間計測
            b_start = time.perf_counter()
            
            #テキスト埋め込みベクトル
            tokens = tokenizer(batch["caption"], max_length=tokenizer.model_max_length, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)
            encoder_hidden_states = text_encoder(tokens, output_hidden_states=True).last_hidden_state.to(device)
            
            #VAEによる潜在変数
            if 'latents' in batch:
                latents = batch['latents'].to(device) * 0.18215 #bucketを使う場合はあらかじめlatentを計算している
            else:
                latents = vae.encode(batch['image'].to(device, dtype=weight_dtype)).latent_dist.sample().to(device) * 0.18215 #正規化
                
            #ミニバッチの拡大
            latents = torch.cat([latents]*config.feature.minibatch_repeat)
            encoder_hidden_states = torch.cat([encoder_hidden_states]*config.feature.minibatch_repeat)
            
            if batch["control"] is not None:
                controls = batch["control"].to(device)
                
                with torch.autocast("cuda",enabled=config.train.amp):
                    pfg_feature = pfg(controls).to(dtype=encoder_hidden_states.dtype)
                    
                encoder_hidden_states = torch.cat([encoder_hidden_states, pfg_feature], dim=1)
                
            #ノイズを生成
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            #画像ごとにstep数を決める
            timesteps = torch.randint(step_range[0], step_range[1], (bsz,), device=latents.device)
            timesteps = timesteps.long()

            #steps数に応じてノイズを付与する
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            #推定ノイズ
            with torch.autocast("cuda",enabled=config.train.amp):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            if config.model.v_prediction:
                noise = noise_scheduler.get_velocity(latents, noise, timesteps)
            
            #顔部分以外をマスクして学ばせない。顔以外をマスクって・・・
            if batch["mask"] is not None:
                mask = batch["mask"].to(device)
                #ミニバッチの拡大
                mask = torch.cat([mask]*config.feature.minibatch_repeat)
                
                noise = noise * mask
                noise_pred = noise_pred * mask
                    
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = loss_ema * 0.9 + loss.item() * 0.1

            #混合精度学習の場合アンダーフローを防ぐために自動でスケーリングしてくれるらしい。
            scaler.scale(loss).backward()
            #勾配降下
            scaler.step(optimizer)
            scaler.update()

            lr_scheduler.step()

            #勾配リセット
            optimizer.zero_grad()
            
            #ステップ更新
            global_steps += 1
            
            #時間計測
            b_end = time.perf_counter()
            time_per_steps = b_end - b_start
            samples_per_time = bsz / time_per_steps
            
            #プログレスバー更新
            logs={"loss":loss_ema,"samples_per_second":samples_per_time,"lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.update(1)
            progress_bar.set_postfix(logs)
            
            #logやsave
            final = total_steps == global_steps
            save(config.model.input_path, global_steps, final, logs, batch, text_encoder, unet, vae, tokenizer, noise_scheduler, network, pfg)
    
        
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

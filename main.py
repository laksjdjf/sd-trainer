import argparse
import torch
from torch.utils.data import DataLoader
import torchvision
import transformers
import diffusers
import os
import tqdm
import numpy as np

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

from utils.dataset import SimpleDataset


###コマンドライン引数#########################################################################
parser = argparse.ArgumentParser(description='Stable Diffusion Finetuner')
parser.add_argument('--model', type=str, required=True, help='pretrained model path')
parser.add_argument('--input', type=str, required=True, help='input path')
parser.add_argument('--output', type=str, required=True, help='output path')
parser.add_argument('--resolution', type=int, default=512, help='resolution of images')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--batch_size', type=float, default=5e-6, help='learning rate')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--amp', action='store_true', help='use auto mixed precision')
args = parser.parse_args()
############################################################################################

def main():
    ###学習準備##############################################################################
    
    #device,dtype
    device = torch.device('cuda')
    weight_dtype = torch.float16 if args.float16 else torch.float32
    
    #モデルのロード、勾配無効や推論モードに移行
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder='tokenizer')
    
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder='text_encoder')
    text_encoder.required_grad_(False)
    text_encoder.eval()
    
    vae = AutoencoderKL.from_pretrained(args.model, subfolder='vae')
    vae.required_grad_(False)
    vae.eval()

    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder='unet')
    try:
        unet.set_use_memory_efficient_attention_xformers(True)
    except:
        print("cant apply xformers. using normal unet !!!")
    unet.required_grad_(True)
    unet.train()
    
    #AMP用のスケーラー
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    #最適化関数
    try:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    except:
        print('cant import bitsandbytes, using regular Adam optimizer')
        optimizer_cls = torch.optim.AdamW
    
    optimizer = optimizer_cls(unet.parameters(),lr=args.lr)
    
    #型の指定とGPUへの移動
    text_encoder.to(device,dtype=weight_dtype)
    vae.to(device,dtype=weight_dtype)
    unet.to(device,dtype=torch.float32) #学習対称はfloat32
    
    #ノイズスケジューラー
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model,
        subfolder='scheduler',
    )
    
    #データローダー
    dataset = SimpleDataset(args.input)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,num_workers=8)

    #学習ループ
    for epoch in range(args.epoch):
        for batch in tqdm(dataloader):
            #テキスト埋め込みベクトル
            tokens = tokenizer(batch["caption"], max_length=tokenizer.model_max_length, padding=True, truncation=True, return_tensors='pt').input_ids
            encoder_hidden_states = text_encoder(token_tensor, output_hidden_states=True).last_hidden_state
            
            #VAEによる潜在変数
            latents = vae.encode(batch['image'].to(device, dtype=weight_dtype)).latent_dist.sample() * 0.18215 #正規化
            
            #ノイズを生成
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            #画像ごとにstep数を決める
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            #steps数に応じてノイズを付与する
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            with torch.autocast("cuda",enabled=args.amp):
                #推定ノイズ
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
            #損失は実ノイズと推定ノイズの誤差である。v_prediction系には対応していない。
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
            #混合精度学習の場合アンダーフローを防ぐために自動でスケーリングしてくれるらしい。
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #勾配降下
            optimizer.zero_grad()
    
        print(f'save in {epoch} epochs')
        pipeline = StableDiffusionPipeline.from_pretrained(
                args.model,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
                scheduler=DDIMScheduler.from_pretrained(args.model, subfolder="scheduler")
            )
        pipeline.save_pretrained(f'{args.output_path}')
if __name__ == "__main__":
    main()


import argparse
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline ,DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from utils.dataset import SimpleDataset


###コマンドライン引数#########################################################################
parser = argparse.ArgumentParser(description='Stable Diffusion Finetuner')
parser.add_argument('--model', type=str, required=True, help='pretrained model path')
parser.add_argument('--dataset', type=str, required=True, help='dataset path')
parser.add_argument('--output', type=str, required=True, help='output path')
parser.add_argument('--image_log', type=str, required=True, help='image log path')
parser.add_argument('--resolution', type=str, default="512,512", help='resolution of images like width,height')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--lr', type=float, default=5e-6, help='learning rate')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--save_n_epochs', type=int, default=5, help='save')
parser.add_argument('--amp', action='store_true', help='use auto mixed precision')
parser.add_argument('--gradient_checkpointing', action='store_true', help='use gradient checkpointing')
args = parser.parse_args()
############################################################################################

def main():
    ###学習準備##############################################################################
    #output pathをつくる。
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(args.image_log):
        os.makedirs(args.image_log)    
        
    #画像サイズ
    size = args.resolution.split(",") 
    size = (int(size[0]),int(size[-1]))
    
    #device,dtype
    device = torch.device('cuda')
    weight_dtype = torch.float16 if torch.float16 else torch.float32
    
    #モデルのロード、勾配無効や推論モードに移行
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder='tokenizer')
    
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder='text_encoder')
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    vae = AutoencoderKL.from_pretrained(args.model, subfolder='vae')
    vae.requires_grad_(False)
    vae.eval()

    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder='unet')
    try:
        unet.set_use_memory_efficient_attention_xformers(True)
    except:
        print("cant apply xformers. using normal unet !!!")
    unet.requires_grad_(True)
    unet.train()
    
    #勾配チェックポイントによるVRAM削減（計算時間増）
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    #AMP用のスケーラー
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    #最適化関数
    try:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
        print('apply AdamW8bit optimizer !')
    except:
        print('cant import bitsandbytes, using regular Adam optimizer !')
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
    dataset = SimpleDataset(args.dataset,size)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,num_workers=2,shuffle=True)
    
    #プログレスバー
    progress_bar = tqdm(range((args.epochs) * len(dataloader)), desc="Total Steps", leave=False)
    loss_ema = 0 
    
    #学習ループ
    for epoch in range(args.epochs):
        for batch in dataloader:
            #テキスト埋め込みベクトル
            tokens = tokenizer(batch["caption"], max_length=tokenizer.model_max_length, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)
            encoder_hidden_states = text_encoder(tokens, output_hidden_states=True).last_hidden_state.to(device)
            
            #VAEによる潜在変数
            latents = vae.encode(batch['image'].to(device, dtype=weight_dtype)).latent_dist.sample().to(device) * 0.18215 #正規化
            
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
            
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = loss_ema * 0.9 + loss.item() * 0.1
            
            #混合精度学習の場合アンダーフローを防ぐために自動でスケーリングしてくれるらしい。
            scaler.scale(loss).backward()
            #勾配降下
            scaler.step(optimizer)
            scaler.update()
            
            #勾配リセット
            optimizer.zero_grad()
            
            #プログレスバー更新
            logs={"loss":loss_ema}
            progress_bar.update(1)
            progress_bar.set_postfix(logs)
        
        #モデルのセーブと検証画像生成
        if epoch % args.save_n_epochs == args.save_n_epochs - 1:
            print(f'save checkpoint !')
            pipeline = StableDiffusionPipeline.from_pretrained(
                    args.model,
                    text_encoder=text_encoder,
                    vae=vae,
                    unet=unet,
                    tokenizer=tokenizer,
                    scheduler=DDIMScheduler.from_pretrained(args.model, subfolder="scheduler"),
                    feature_extractor = None,
                    safety_checker = None
            )
            with torch.autocast('cuda', enabled=args.amp):    
                image = pipeline(batch["caption"][0],width=size[0],height=size[1]).images[0]
            image.save(os.path.join(args.image_log,f'image_log_{str(epoch).zfill(3)}.png'))
            pipeline.save_pretrained(f'{args.output}')
            del pipeline
        
if __name__ == "__main__":
    main()


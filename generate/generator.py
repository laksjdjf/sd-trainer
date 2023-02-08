import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np
from tqdm import tqdm

class SDGenerator:
    def __init__(self,model_id, dtype=torch.float32, device="cuda"):
        '''
        model_id：diffusersモデルのパス/huggingfaceのリポジトリ
        dtype, device:わかるやろ
        '''
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder='text_encoder').eval().to(device, dtype=dtype)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder='vae').eval().to(device, dtype=dtype)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder='unet').eval().to(device, dtype=dtype)
        try:
            self.unet.set_use_memory_efficient_attention_xformers(True)
            print("xformersを適用しますた。")
        except:
            print("xformersがインストールされていないようなのでそのままやります。")
            
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.dtype = dtype
        self.device = device

    def encode_prompts(self, prompts):
        '''
        プロンプトをとーくんにしてtext_encoderの隠れ状態を出力する。
        promptsはリストであることを前提とする。
        '''
        with torch.no_grad():
            tokens = self.tokenizer(prompts, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors='pt').input_ids.to(self.device)
            embs = self.text_encoder(tokens, output_hidden_states=True).last_hidden_state.to(self.device, dtype = self.dtype)
        return embs

    def decode_latents(self, latents):
        '''
        潜在変数からPillowに変換
        '''
        latents = 1 / 0.18215 * latents
        with torch.no_grad():    
            images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def __call__(self,prompts, negative_prompts, height:int = 896, width:int = 640, guidance_scale:float = 7.0, num_inference_steps:int = 50):
        '''
        prompts, negative_promptsは文字列か文字列のリスト。
        negative_promptsが1個の場合はpromptsの数に合わせる。
        '''
        if type(prompts) == str:
            prompts = [prompts]
        if type(negative_prompts) == str:
            negative_prompts = [negative_prompts]
        if len(negative_prompts) == 1:
            negative_prompts = negative_prompts * len(prompts)

        assert len(prompts) == len(negative_prompts), "プロンプトとネガティブプロンプトの数が一致していません"


        #プロンプト、ネガティブプロンプトのtext_encoder出力
        text_embs = self.encode_prompts(prompts+negative_prompts)

        #スケジューラーのtimestepを設定
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        #初期ノイズ
        latents = torch.randn((len(prompts), 4, height // 8, width // 8), device = self.device, dtype = self.dtype)
        latents = latents * self.scheduler.init_noise_sigma

        for i,t in tqdm(enumerate(timesteps)):
            #入力を作成
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            #推定ノイズ
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input,t,encoder_hidden_states=text_embs).sample
            
            #ネガティブプロンプト版CFG
            noise_pred_text, noise_pred_negative= noise_pred.chunk(2)
            noise_pred = noise_pred_negative + guidance_scale * (noise_pred_text - noise_pred_negative)

            #推定ノイズからノイズを取り除いたlatentsを求める
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        images = self.decode_latents(latents)
        return images

import torch
import os
from utils.generate import StableDiffusionGenerator
from utils.functions import default
import wandb

# Saveクラス：
# __init__には出力先：outputと、1エポックごとのステップ数steps_par_epochを引数に取ることが必要。
# __call__にはSaveと同様の引数を取ることが必要。


# 保存先のディレクトリ
DIRECTORIES = [
    "trained",
    "trained/models",
    "trained/networks",
    "trained/pfg",
    "trained/controlnet",
    "trained/embeddings",
    "trained/ip_adapter",
]

class Save:
    def __init__(
            self,
            config,
            text_model,
            steps_per_epoch: int,
            wandb_name: str = "sd-trainer",
            image_logs: str = "image_logs",
            num_images: int = 4,
            resolution: str = "640,896",
            save_n_epochs: int = 1,
            save_n_steps: int = None,
            over_write: bool = True,
            prompt: str = None,
            negative_prompt: str = "",
            seed=4545,
        ):

        # save_n_epochsよりsave_n_stepsが優先となる。
        # wandbを使うときはimage_logsに画像を保存しない。

        self.config = config
        self.output = config.model.output_name
        self.image_logs = image_logs
        self.clip_skip = default(self.config.model, "clip_skip", -1)
        if self.image_logs is not None:
            os.makedirs(self.image_logs, exist_ok=True)

        for directory in DIRECTORIES:
            os.makedirs(directory, exist_ok=True)

        self.num_images = num_images
        self.wandb = wandb_name

        if self.wandb:
            self.run = wandb.init(project=self.wandb, name=self.output, dir="wandb")

        self.save_n_steps = save_n_steps if save_n_steps is not None else save_n_epochs * steps_per_epoch

        print(f"セーブ間隔：{self.save_n_steps}ステップごと")

        self.resolution = resolution.split(",")
        self.resolution = (int(self.resolution[0]), int(self.resolution[-1]))  # 長さが1の場合同じ値になる

        self.over_write = over_write

        self.prompt = prompt
        self.negative_prompt = negative_prompt if negative_prompt is not None else ""

        self.sdxl = text_model.text_encoder_2 is not None

        # ネガティブプロンプトの事前計算
        text_device = text_model.text_encoder.device
        text_model.to("cuda")
        with torch.no_grad():
            tokens, tokens_2, empty_text = text_model.tokenize([self.negative_prompt])
            self.uncond_hidden_state, self.uncond_pooled_output = text_model(tokens, tokens_2, empty_text)
            self.uncond_hidden_state = self.uncond_hidden_state.detach().float().cpu()
            self.uncond_pooled_output = None if self.uncond_pooled_output is None else self.uncond_pooled_output.detach().float().cpu()
        text_model.to(text_device)

        self.seed = seed

    @torch.no_grad()
    def __call__(self, steps, final, logs, batch, text_model, unet, vae, scheduler, network=None, pfg=None, controlnet=None, ip_adapter=None):
        if self.wandb:
            self.run.log(logs, step=steps)
            
        if steps % self.save_n_steps == 0 or final:
            print(f'チェックポイントをセーブするよ!')
            pipeline = StableDiffusionGenerator(
                unet=unet,
                vae=vae,
                text_model=text_model,
                scheduler_config=scheduler.config,
            )

            filename = f"{self.output}" if self.over_write else f"{self.output}_{steps}"

            if self.config.train.train_unet:
                pipeline.save_pretrained(os.path.join("trained/models", filename))

            if network is not None and self.config.network.train:
                network.save_weights(os.path.join("trained/networks", filename))

            if controlnet is not None and self.config.controlnet.train:
                controlnet.save_pretrained(os.path.join("trained/controlnet", filename))

            if pfg is not None and self.config.pfg.train:
                pfg.save_weights(os.path.join("trained/pfg", filename + '.pt'))
            
            if text_model.textual_inversion:
                text_model.save_embeddings(os.path.join("trained/embeddings", filename))

            if ip_adapter is not None and self.config.ip_adapter.train:
                ip_adapter.save_ip_adapter(os.path.join("trained/ip_adapter", filename + '.bin'))

            # 検証画像生成
            torch.cuda.empty_cache()
            vae_device = pipeline.vae.device
            vae.to("cuda")
            with torch.autocast('cuda',dtype=torch.bfloat16):
                # バッチサイズがnum_imagesより小さい場合はバッチサイズに合わせる
                if "encoder_hidden_states" in batch:
                    text_embeds = (batch["encoder_hidden_states"], batch["pooled_outputs"])
                    num_images = min(text_embeds[0].shape[0], self.num_images)
                    prompt = ""
                else:
                    prompts = batch["captions"][:self.num_images] if self.prompt is None else [self.prompt] * self.num_images
                    num_images = min(len(prompts), self.num_images)
                    text_embeds = None
                
                images = []
                for i in range(num_images):
                    if text_embeds is not None:
                        uncond_hidden_state = self.uncond_hidden_state.clone()
                        uncond_pooled_output = self.uncond_pooled_output.clone()
                        encoder_hidden_states = torch.cat([text_embeds[0][i].unsqueeze(0), uncond_hidden_state]).to(unet.device, dtype=unet.dtype)
                        pooled_outputs = torch.cat([text_embeds[1][i].unsqueeze(0), uncond_pooled_output]).to(unet.device, dtype=unet.dtype)
                        text_embed = (encoder_hidden_states, pooled_outputs)
                        prompt = ""
                    else:
                        prompt = prompts[i]
                        text_embed = None

                    if pfg is not None:
                        pfg_feature = pfg(batch["pfg"][i].unsqueeze(0).to("cuda"))
                    else:
                        pfg_feature = None

                    if controlnet is not None:
                        guide_image = batch["control"][i].unsqueeze(0).to(controlnet.device, dtype=controlnet.dtype)
                        self.resolution = (guide_image.shape[3], guide_image.shape[2])  # width, height
                    else:
                        guide_image = None

                    if ip_adapter is not None:
                        cond = ip_adapter.get_image_embeds(batch["image_embeds"][i].unsqueeze(0).to("cuda"))
                        uncond = ip_adapter.get_image_embeds(torch.zeros_like(batch["image_embeds"][i]).unsqueeze(0).to("cuda"))
                        ip_adapter.set_ip_hidden_states(torch.cat([cond, uncond]))
                    else:
                        ip_adapter_feature = None
                        
                    if 'control' in batch and hasattr(network, "set_cond_image"):
                        network.set_cond_image(batch["control"][i].unsqueeze(0).to("cuda"))
                        self.resolution = (batch["control"].shape[3], batch["control"].shape[2])  # width, height
                    
                    image = pipeline.generate(
                        prompt,
                        self.negative_prompt,
                        width=self.resolution[0],
                        height=self.resolution[1],
                        pfg_feature=pfg_feature,
                        controlnet=controlnet,
                        guide_image=guide_image,
                        text_embeds=text_embed,
                        seed=self.seed + i,
                    )[0]
                    if self.wandb:
                        images.append(wandb.Image(image, caption=prompt))
                    else:
                        image.save(os.path.join(self.image_logs, f'image_log_{str(steps).zfill(6)}_{i}.png'))
            vae.to(vae_device)
            if self.wandb:
                self.run.log({'images': images}, step=steps)

            del pipeline
            torch.cuda.empty_cache()

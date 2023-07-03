import torch
import os
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from utils.generate import WrapStableDiffusionPipeline
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
    "trained/controlnet"
]

class Save:
    def __init__(self,
                 config,
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
        self.negative_prompt = negative_prompt

        self.seed = seed

    @torch.no_grad()
    def __call__(self, steps, final, logs, batch, text_encoder, unet, vae, tokenizer, scheduler, network=None, pfg=None, controlnet=None):
        if self.wandb:
            self.run.log(logs, step=steps)
        if steps % self.save_n_steps == 0 or final:
            print(f'チェックポイントをセーブするよ!')
            pipeline = WrapStableDiffusionPipeline(
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
                scheduler=scheduler,
                feature_extractor=None,
                safety_checker=None
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

            # 検証画像生成
            with torch.autocast('cuda'):
                # バッチサイズがnum_imagesより小さい場合はバッチサイズに合わせる
                num_images = min(len(batch["captions"]), self.num_images)
                images = []
                for i in range(num_images):
                    prompt = batch["captions"][i] if self.prompt is None else self.prompt

                    if pfg is not None:
                        pfg_feature = pfg(batch["pfg"][i].unsqueeze(0).to("cuda"))
                    else:
                        pfg_feature = None
                        
                    if controlnet is not None:
                        guide_image = batch["control"][i].unsqueeze(0).to(controlnet.device,dtype =controlnet.dtype)
                        self.resolution = (guide_image.shape[3], guide_image.shape[2]) #width, height
                    else:
                        guide_image = None

                    image = pipeline.generate(prompt,
                                              self.negative_prompt,
                                              width=self.resolution[0],
                                              height=self.resolution[1],
                                              pfg_feature=pfg_feature,
                                              controlnet=controlnet,
                                              guide_image = guide_image,
                                              seed=self.seed + i
                                              )[0]
                    if self.wandb:
                        images.append(wandb.Image(image, caption=prompt))
                    else:
                        image.save(os.path.join(self.image_logs, f'image_log_{str(steps).zfill(6)}_{i}.png'))
                    
                    
            if self.wandb:
                self.run.log({'images': images}, step=steps)

            del pipeline
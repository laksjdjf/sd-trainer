import torch
import os
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import wandb

#Saveクラス：
#__init__には出力先：outputと、1エポックごとのステップ数steps_par_epochを引数に取ることが必要。
#__call__にはSaveと同様の引数を取ることが必要。


class Save:
    def __init__(self,
                 output:str,
                 steps_par_epoch:int, 
                 wandb_name:str = "sd-trainer", 
                 image_logs:str = "image_logs", 
                 num_images:int = 4, 
                 resolution:str = "640,896", 
                 save_n_epochs:int = 1, 
                 save_n_steps:int = None, 
                 over_write:bool = True, 
                 prompt:str = None, 
                 negative_prompt:str = "", 
                 seed = 4545,
                ):
        '''
        save_n_epochsよりsave_n_stepsが優先となる。
        wandbを使うときはimage_logsに画像を保存しない。
        '''
        self.output = output
        
        self.image_logs = image_logs
        self.num_images = num_images
        
        self.wandb = wandb_name
        
        if self.wandb:
            self.run = wandb.init(project=self.wandb, name=output, dir=os.path.join(output,'wandb'))
        
        self.save_n_steps = save_n_steps if save_n_steps is not None else save_n_epochs * steps_par_epoch
        
        print(f"セーブ間隔：{self.save_n_steps}ステップごと")
        
        self.resolution = resolution.split(",") 
        self.resolution = (int(self.resolution[0]),int(self.resolution[-1])) #長さが1の場合同じ値になる
        
        self.over_write = over_write
        
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        
        self.seed = seed
    
    def __call__(self, model_id, steps, logs, batch, text_encoder, unet, vae, tokenizer, network = None):
        if self.wandb:
            self.run.log(logs, step=steps)
        if steps % self.save_n_steps == 0:
            print(f'チェックポイントをセーブするよ!')
            pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    text_encoder=text_encoder,
                    vae=vae,
                    unet=unet,
                    tokenizer=tokenizer,
                    feature_extractor = None,
                    safety_checker = None
            )
            
            filename = f"{self.output}" if self.over_write else f"{self.output}_{steps}"

            if network is not None:
                network.save_weights(filename + '.pt')
            else:
                pipeline.save_pretrained(filename)

            #検証画像生成
            with torch.autocast('cuda'):
                num = min(len(batch["caption"]), self.num_images) #基本4枚だがバッチサイズ次第
                images = []
                generator = torch.Generator("cuda").manual_seed(self.seed)
                for i in range(num):
                    prompt = batch["caption"][i] if self.prompt is None else self.prompt
                    if batch["control"] is not None:
                        network.set_input(batch["control"][i].unsqueeze(0).to("cuda"))
                    image = pipeline(prompt,width=self.resolution[0],height=self.resolution[1],negative_prompt=self.negative_prompt,generator=generator).images[0]
                    if self.wandb:    
                        images.append(wandb.Image(image,caption=prompt))
                    else:
                        images.append(image)
                        
            if self.wandb:
                self.run.log({'images': images}, step=steps)
            else:
                [image.save(os.path.join(self.image_log,f'image_log_epoch_{str(steps).zfill(6)}_{i}.png')) for i,image in enumerate(images)]

            del pipeline

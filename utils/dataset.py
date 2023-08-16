from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import random
import numpy as np
from transformers import CLIPTokenizer
from utils.model import TextModel
from typing import Optional

class BaseDataset(Dataset):
    def __init__(
            self,
            config,
            text_model: TextModel,
            path: str,
            metadata: str,
            original_size: Optional[str] = None,
            latent: Optional[str] = "latents",
            caption: Optional[str] = "captions",
            image: Optional[str] = None,
            text_emb: Optional[str] = None,
            mask: Optional[str] = None,
            pfg: Optional[str] = None,
            control: Optional[str] = None,
            prompt: Optional[str] = None,
            prefix: str = "",
            shuffle: bool = False,
            ucg: float = 0.0
        ):

        with open(os.path.join(path, metadata), "r") as f:
            self.bucket2file = json.load(f)

        if original_size is not None:
            with open(os.path.join(path, original_size), "r") as f:
                self.original_size = json.load(f)
        else:
            self.original_size = {}

        self.path = path
        self.batch_size = config.train.batch_size
        self.minibatch_repeat = config.feature.minibatch_repeat
        self.minibatch_size = self.batch_size // self.minibatch_repeat
        self.text_model = text_model
        self.latent = latent
        self.caption = caption
        self.image = image
        self.text_emb = text_emb
        self.mask = mask
        self.pfg = pfg
        self.control = control
        self.prompt = prompt  # 全ての画像のcaptionをpromptにする
        self.prefix = prefix  # captionのprefix
        self.shuffle = shuffle  # バッチの取り出し方をシャッフルするかどうか（データローダー側でシャッフルした方が良い＾＾）
        self.ucg = ucg  # captionをランダムにする空文にする確率

        # 空文の埋め込みを事前に計算しておく
        if self.ucg > 0.0 and self.text_emb:
            text_device = self.text_model.device
            self.text_model.to("cuda")
            tokens, tokens_2, empty_text = self.text_model.tokenize([""])
            with torch.no_grad():
                self.uncond_hidden_state, self.uncond_pooled_output = self.text_model.get_text_embeddings(tokens, tokens_2, empty_text)
            self.uncond_hidden_state.detach().float().cpu()
            self.uncond_pooled_output.detach().float().cpu()
            self.text_model.to(text_device)

        self.init_batch_samples()

    def __len__(self):
        return len(self.batch_samples)

    def __getitem__(self, i):
        if i == 0 and self.shuffle:
            self.init_batch_samples()

        batch = {}
        samples = self.batch_samples[i]

        if self.image:
            images = self.get_images(samples, self.image if isinstance(self.image, str) else "images")
            batch["images"] = torch.cat([images]*self.minibatch_repeat, dim=0)
            target_height, target_width = images.shape[2], images.shape[3]
        else:
            latents = self.get_latents(samples, self.latent)
            batch["latents"] = torch.cat([latents]*self.minibatch_repeat, dim=0)
            target_height, target_width = latents.shape[2]*8, latents.shape[3]*8

        batch["size_condition"] = self.get_size_condition(samples, target_height, target_width)

        if self.text_emb:
            encoder_hidden_states, pooled_outputs = self.get_text_embeddings(samples, self.text_emb if isinstance(self.text_emb, str) else "text_emb")
            batch["encoder_hidden_states"] = torch.cat([encoder_hidden_states]*self.minibatch_repeat, dim=0)
            batch["pooled_outputs"] = torch.cat([pooled_outputs]*self.minibatch_repeat, dim=0)
        else:
            captions = self.get_captions(samples, self.caption)
            batch["captions"] = captions * self.minibatch_repeat
        
        if self.mask:
            masks = self.get_masks(samples, self.mask if isinstance(self.mask, str) else "mask")
            batch["mask"] = torch.cat([masks]*self.minibatch_repeat, dim=0)
        if self.pfg:
            pfg = self.get_pfg(samples, self.pfg if isinstance(self.pfg, str) else "pfg")
            batch["pfg"] = torch.cat([pfg]*self.minibatch_repeat, dim=0)
        if self.control:
            control = self.get_control(samples, self.control if isinstance(self.control, str) else "control")
            batch["control"] = torch.cat([control]*self.minibatch_repeat, dim=0)

        return batch

    # バッチの取り出し方を初期化するメソッド
    def init_batch_samples(self):
        self.batch_samples = []
        for key in self.bucket2file:
            random.shuffle(self.bucket2file[key])
            self.batch_samples.extend([self.bucket2file[key][i:i+self.minibatch_size]
                                      for i in range(0, len(self.bucket2file[key]), self.minibatch_size)])
        random.shuffle(self.batch_samples)

    def get_images(self, samples, dir="images"):
        images = []
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        for sample in samples:
            image = Image.open(os.path.join(self.path, dir, sample + ".png")).convert("RGB")
            image = transform(image)
            images.append(image)

        images = torch.stack(images)
        images = images.to(memory_format=torch.contiguous_format).float()
        return images
        
    def get_latents(self, samples, dir="latents"):
        latents = torch.stack([torch.tensor(np.load(os.path.join(self.path, dir, sample + ".npy"))) for sample in samples])
        latents = latents.to(memory_format=torch.contiguous_format).float()  # これなに
        return latents

    def get_captions(self, samples, dir="captions"):
        captions = []
        for sample in samples:
            if self.prompt is None:
                with open(os.path.join(self.path, dir, sample + ".caption"), "r") as f:
                    caption = self.prefix + f.read()
            else:
                caption = self.prompt

            if random.random() < self.ucg:
                caption = ""
            captions.append(caption)
        return captions
    
    def get_size_condition(self, samples, target_height, target_width):
        size_condition = []
        for sample in samples:
            if sample in self.original_size:
                original_width = self.original_size[sample]["original_width"]
                original_height = self.original_size[sample]["original_height"]
            
                original_ratio = original_width / original_height
                target_ratio = target_width / target_height

                if original_ratio > target_ratio: # 横長の場合
                    resize_ratio = target_width / original_width # 横幅を合わせる
                    resized_height = original_height * resize_ratio # 縦をリサイズ
                    crop_top = (target_height - resized_height) // 2 # 上部の足りない分がcrop_top
                    crop_left = 0
                else:
                    resize_ratio = target_height / original_height
                    resize_width = original_width * resize_ratio 
                    crop_top = 0
                    crop_left = (target_width - resize_width) // 2
            else:
                original_width, original_height = target_width, target_height
                crop_top = 0
                crop_left = 0
            size_list = [original_height, original_width, crop_top, crop_left, target_height, target_width]    
            size_condition.append(torch.tensor(size_list))
        return torch.stack(size_condition)
    
    def get_text_embeddings(self, samples, dir="text_emb"):
        if random.random() < self.ucg:
            encoder_hidden_states = self.uncond_hidden_state.repeat(len(samples), 1, 1)
            pooled_outputs = self.uncond_pooled_output.repeat(len(samples), 1)
            return encoder_hidden_states, pooled_outputs

        encoder_hidden_states = torch.stack([
            torch.tensor(np.load(os.path.join(self.path, dir, sample + ".npz"))["encoder_hidden_state"])
            for sample in samples
        ])
        encoder_hidden_states.to(memory_format=torch.contiguous_format).float()
        
        pooled_outputs = torch.stack([
            torch.tensor(np.load(os.path.join(self.path, dir, sample + ".npz"))["pooled_output"])
            for sample in samples
        ])
        pooled_outputs.to(memory_format=torch.contiguous_format).float()
        return encoder_hidden_states, pooled_outputs

    def get_masks(self, samples, dir="mask"):
        masks = torch.stack([
            torch.tensor(np.load(os.path.join(self.path, dir, sample + ".npz"))["arr_0"]).unsqueeze(0).repeat(4, 1, 1)
            for sample in samples
        ])
        masks.to(memory_format=torch.contiguous_format).float()
        return masks

    def get_pfg(self, samples, dir="pfg"):
        pfg = torch.stack([
            torch.tensor(np.load(os.path.join(self.path, dir, sample + ".npz"))["controll"]).unsqueeze(0)
            for sample in samples
        ])
        pfg.to(memory_format=torch.contiguous_format).float()
        return pfg

    def get_control(self, samples, dir="control"):
        images = []
        for sample in samples:
            image = Image.open(os.path.join(self.path, dir, sample + f".png"))
            image = np.array(image.convert("RGB"))
            image = image[None, :]
            images.append(image)
        images_tensor = np.concatenate(images, axis=0)
        images_tensor = np.array(images_tensor).astype(np.float32) / 255.0
        images_tensor = images_tensor.transpose(0, 3, 1, 2)
        images_tensor = torch.from_numpy(images_tensor).to(memory_format=torch.contiguous_format).float()
        return images_tensor

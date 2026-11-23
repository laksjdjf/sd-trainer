from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import logging
import random
import numpy as np
from typing import Optional
from modules.text.base import BaseTextOutput

logger = logging.getLogger("データセットちゃん")

class BaseDataset(Dataset):
    def __init__(
        self,
        text_model,
        batch_size: int,
        path: str,
        metadata: str="buckets.json",
        original_size: Optional[str] = None,
        latent: Optional[str] = "latents",
        caption: Optional[str] = "captions",
        image: Optional[str] = None,
        text_emb: Optional[str] = None,
        control: Optional[str] = None,
        mask: Optional[str] = None,
        prompt: Optional[str] = None,
        prefix: str = "",
        shuffle: bool = False,
        ucg: float = 0.0,
        model_type: str = "sdxl"
    ):

        with open(os.path.join(path, metadata), "r") as f:
            self.bucket2file = json.load(f)

        if original_size is not None:
            with open(os.path.join(path, original_size), "r") as f:
                self.original_size = json.load(f)
        else:
            self.original_size = {}

        from modules.model_registry import get_model_spec
        spec = get_model_spec(model_type)
        self.latent_scale_factor = spec.latent_scale_factor
        self.image_mode = spec.image_mode
        self.model_type = model_type
        self.path = path
        self.batch_size = batch_size
        self.text_model = text_model
        self.latent = latent
        self.caption = caption
        self.image = image
        self.text_emb = text_emb
        self.control = control
        self.mask = mask
        self.prompt = prompt  # 全ての画像のcaptionをpromptにする
        self.prefix = prefix  # captionのprefix
        self.shuffle = shuffle  # Trueならエポックごとにバッチの組み分けを変える(worker複製前にmain側でinit_batch_samplesを呼ぶ)
        self.ucg = ucg  # captionをランダムにする空文にする確率

        self.init_batch_samples()
        logger.info(f"データセットを作ったよ！")

    def __len__(self):
        return len(self.batch_samples)

    def __getitem__(self, i):
        batch = {}
        samples = self.batch_samples[i]

        if self.image:
            batch["images"] = self.get_images(samples, self.image if isinstance(self.image, str) else "images")
            target_height, target_width = batch["images"].shape[2:]
        else:
            batch["latents"] = self.get_latents(samples, self.latent)
            target_height, target_width = (n * self.latent_scale_factor for n in batch["latents"].shape[-2:])

        batch["size_condition"] = self.get_size_condition(samples, target_height, target_width)

        if self.text_emb:
            text_output = self.get_text_output(samples, self.text_emb if isinstance(self.text_emb, str) else "text_emb")
            batch["encoder_hidden_states"] = text_output.encoder_hidden_states
            if text_output.pooled_output is not None:
                batch["pooled_outputs"] = text_output.pooled_output
            if text_output.attention_mask is not None:
                batch["attention_mask"] = text_output.attention_mask
        else:
            batch["captions"] = self.get_captions(samples, self.caption)

        if self.control:
            batch["controlnet_hint"] = self.get_control(samples, self.control if isinstance(self.control, str) else "control")

        if self.mask:
            batch["mask"] = self.get_masks(samples, self.mask if isinstance(self.mask, str) else "mask")

        return batch

    # バッチの取り出し方を初期化するメソッド
    def init_batch_samples(self):
        self.batch_samples = []
        for key in self.bucket2file:
            random.shuffle(self.bucket2file[key])
            self.batch_samples.extend([self.bucket2file[key][i:i+self.batch_size]
                                      for i in range(0, len(self.bucket2file[key]), self.batch_size)])
        random.shuffle(self.batch_samples)

    def get_images(self, samples, dir="images"):
        images = []
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        for sample in samples:
            image = Image.open(os.path.join(self.path, dir, sample + ".png")).convert(getattr(self, "image_mode", "RGB"))
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
    
    def get_text_output(self, samples, dir="text_emb"):
        outputs = []
        for sample in samples:
            if random.random() < self.ucg:
                outputs.append(self.text_model.uncond_output.clone())
                continue
            with np.load(os.path.join(self.path, dir, sample + ".npz")) as f:
                hidden = torch.from_numpy(f["encoder_hidden_state"].copy()).float().unsqueeze(0)
                pooled = torch.from_numpy(f["pooled_output"].copy()).float().unsqueeze(0) if "pooled_output" in f else None
                mask = torch.from_numpy(f["attention_mask"].copy()).bool().unsqueeze(0) if "attention_mask" in f else None
                if getattr(self, "model_type", "sdxl") == "qwenimage21" and mask is None:
                    raise ValueError("Qwen-Image-2.1 text caches require attention_mask; regenerate them.")
                if getattr(self, "model_type", "sdxl") == "moss_image" and mask is None:
                    raise ValueError("MOSS-Image text caches require attention_mask; regenerate them.")
                outputs.append(BaseTextOutput(hidden, pooled, mask))
        return BaseTextOutput.cat(outputs)

    def get_text_embeddings(self, samples, dir="text_emb"):
        # Backwards-compatible helper for dataset subclasses.
        output = self.get_text_output(samples, dir)
        return output.encoder_hidden_states, output.pooled_output

    def get_control(self, samples, dir="control"):
        images = []
        transform = transforms.ToTensor()
        for sample in samples:
            image = Image.open(os.path.join(self.path, dir, sample + f".png")).convert("RGB")
            images.append(transform(image))
        images_tensor = torch.stack(images).to(memory_format=torch.contiguous_format).float()
        return images_tensor

    def get_masks(self, samples, dir="mask"):
        masks = torch.stack([
            torch.tensor(np.load(os.path.join(self.path, dir, sample + ".npz"))["arr_0"]).unsqueeze(0)
            for sample in samples
        ])
        masks = masks.to(memory_format=torch.contiguous_format).float()
        return masks

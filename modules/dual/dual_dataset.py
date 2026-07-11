from modules.dataset import BaseDataset
import os
import json
import logging
from typing import Optional

logger = logging.getLogger("データセットちゃん")

class DualDataset(BaseDataset):
    def __init__(
        self,
        text_model,
        batch_size: int,
        path: str,
        metadata: str="buckets.json",
        original_size: Optional[str] = None,
        latent_w: Optional[str] = "latents_w",
        latent_l: Optional[str] = "latents_l",
        caption: Optional[str] = "captions",
        caption_l: Optional[str] = "captions_l",
        image_w: Optional[str] = None,
        image_l: Optional[str] = None,
        text_emb: Optional[str] = None,
        text_emb_l: Optional[str] = None,
        control: Optional[str] = None,
        mask: Optional[str] = None,
        prompt_w: Optional[str] = None,
        prompt_l: Optional[str] = None,
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
        self.batch_size = batch_size
        self.text_model = text_model
        self.latent_w = latent_w
        self.latent_l = latent_l
        self.caption = caption
        self.caption_l = caption_l
        self.image_w = image_w
        self.image_l = image_l
        self.text_emb = text_emb
        self.text_emb_l = text_emb_l
        self.control = control
        self.mask = mask
        self.prompt_w = prompt_w  # 全ての画像のcaptionをpromptにする
        self.prompt_l = prompt_l
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

        if self.image_w:
            batch["images_w"] = self.get_images(samples, self.image_w if isinstance(self.image_w, str) else "images_w")
            target_height, target_width = batch["images_w"].shape[2:]
        else:
            batch["latents_w"] = self.get_latents(samples, self.latent_w)
            target_height, target_width = batch["latents_w"].shape[2]*8, batch["latents_w"].shape[3]*8

        if self.image_l:
            batch["images_l"] = self.get_images(samples, self.image_l if isinstance(self.image_l, str) else "images_l")
        else:
            batch["latents_l"] = self.get_latents(samples, self.latent_l)

        batch["size_condition"] = self.get_size_condition(samples, target_height, target_width)

        if self.text_emb:
            batch["encoder_hidden_states"], batch["pooled_outputs"] = self.get_text_embeddings(samples, self.text_emb if isinstance(self.text_emb, str) else "text_emb")
        else:
            self.prompt = self.prompt_w
            batch["captions"] = self.get_captions(samples, self.caption)

        if self.text_emb_l:
            batch["encoder_hidden_states_l"], batch["pooled_outputs_l"] = self.get_text_embeddings(samples, self.text_emb_l if isinstance(self.text_emb_l, str) else "text_emb_l")
        else:
            if self.caption_l:
                self.prompt = self.prompt_l
                batch["captions_l"] = self.get_captions(samples, self.caption_l)

        if self.control:
            batch["controlnet_hint"] = self.get_control(samples, self.control if isinstance(self.control, str) else "control")

        if self.mask:
            batch["mask"] = self.get_masks(samples, self.mask if isinstance(self.mask, str) else "mask")

        return batch
    
class TripleDataset(BaseDataset):
    def __init__(
        self,
        text_model,
        batch_size: int,
        path: str,
        metadata: str="buckets.json",
        original_size: Optional[str] = None,
        latent_w: Optional[str] = "latents_w",
        latent_l: Optional[str] = "latents_l",
        latent_n: Optional[str] = "latents_n",
        caption: Optional[str] = "captions",
        image_w: Optional[str] = None,
        image_l: Optional[str] = None,
        image_n: Optional[str] = None,
        text_emb: Optional[str] = None,
        control: Optional[str] = None,
        mask: Optional[str] = None,
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
        self.batch_size = batch_size
        self.text_model = text_model
        self.latent_w = latent_w
        self.latent_l = latent_l
        self.latent_n = latent_n
        self.caption = caption
        self.image_w = image_w
        self.image_l = image_l
        self.image_n = image_n
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

        if self.image_w:
            batch["images_w"] = self.get_images(samples, self.image_w if isinstance(self.image_w, str) else "images_w")
            target_height, target_width = batch["images_w"].shape[2:]
        else:
            batch["latents_w"] = self.get_latents(samples, self.latent_w)
            target_height, target_width = batch["latents_w"].shape[2]*8, batch["latents_w"].shape[3]*8

        if self.image_l:
            batch["images_l"] = self.get_images(samples, self.image_l if isinstance(self.image_l, str) else "images_l")
        else:
            batch["latents_l"] = self.get_latents(samples, self.latent_l)

        if self.image_n:
            batch["images_n"] = self.get_images(samples, self.image_n if isinstance(self.image_n, str) else "images_n")
        else:
            batch["latents_n"] = self.get_latents(samples, self.latent_n)

        batch["size_condition"] = self.get_size_condition(samples, target_height, target_width)

        if self.text_emb:
            batch["encoder_hidden_states"], batch["pooled_outputs"] = self.get_text_embeddings(samples, self.text_emb if isinstance(self.text_emb, str) else "text_emb")
        else:
            batch["captions"] = self.get_captions(samples, self.caption)

        if self.control:
            batch["controlnet_hint"] = self.get_control(samples, self.control if isinstance(self.control, str) else "control")

        if self.mask:
            batch["mask"] = self.get_masks(samples, self.mask if isinstance(self.mask, str) else "mask")

        return batch
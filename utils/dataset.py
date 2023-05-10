from PIL import Image
import os
import torch
from torch.utils.data import Dataset
import json
import random
import numpy as np
from transformers import CLIPTokenizer
from torchvision import transforms
# dataset:tokenizer, path, batch_size, minibatch_repeat, metadataを引数に取ることが必須。
# batchは"latents"か"images"のどちらかと、"captions"が必須。

class BaseDataset(Dataset):
    def __init__(self,
                 tokenizer: CLIPTokenizer,
                 batch_size: int,
                 minibatch_repeat: int,
                 path: str,
                 metadata: str,
                 mask: bool = False,
                 pfg: bool = False,
                 control: bool = False,
                 prompt: str = None,
                 prefix: str = "",
                 shuffle: bool = True,
                 ucg: False = 0.0):

        with open(os.path.join(path, metadata), "r") as f:
            self.bucket2file = json.load(f)
        self.path = path
        self.batch_size = batch_size
        self.minibatch_repeat = minibatch_repeat
        self.minibatch_size = self.batch_size // self.minibatch_repeat
        self.tokenizer = tokenizer # このクラスでは使っていない
        self.mask = mask
        self.pfg = pfg
        self.control = control 
        self.prompt = prompt # 全ての画像のcaptionをpromptにする
        self.prefix = prefix # captionのprefix
        self.shuffle = shuffle # バッチの取り出し方をシャッフルするかどうか（データローダー側でシャッフルした方が良い＾＾）
        self.ucg = ucg # captionをランダムにする空文にする確率

        self.init_batch_samples()

    def __len__(self):
        return len(self.batch_samples)

    def __getitem__(self, i):
        if i == 0 and self.shuffle:
            self.init_batch_samples()

        batch = {}
        samples = self.batch_samples[i]

        latents = self.get_latents(samples)
        batch["latents"] = torch.cat([latents]*self.minibatch_repeat, dim=0)
        captions = self.get_captions(samples)
        batch["captions"] = captions * self.minibatch_repeat

        if self.mask:
            masks = self.get_masks(samples)
            batch["masks"] = torch.cat([masks]*self.minibatch_repeat, dim=0)
        if self.pfg:
            pfg = self.get_pfg(samples)
            batch["pfg"] = torch.cat([pfg]*self.minibatch_repeat, dim=0)
        if self.control:
            control = self.get_control(samples)
            batch["control"] = torch.cat([control]*self.minibatch_repeat, dim=0)

        return batch

    # バッチの取り出し方を初期化するメソッド
    def init_batch_samples(self):
        random.seed()
        self.batch_samples = []
        for key in self.bucket2file:
            random.shuffle(self.bucket2file[key])
            self.batch_samples.extend([self.bucket2file[key][i:i+self.minibatch_size]
                                      for i in range(0, len(self.bucket2file[key]), self.minibatch_size)])
        random.shuffle(self.batch_samples)

    def get_latents(self, samples):
        latents = torch.stack([torch.tensor(np.load(os.path.join(self.path, "latents", sample + ".npy"))) for sample in samples])
        latents = latents.to(memory_format=torch.contiguous_format).float()  # これなに
        return latents

    def get_captions(self, samples):
        captions = []
        for sample in samples:
            if self.prompt is None:
                with open(os.path.join(self.path, "captions", sample + ".caption"), "r") as f:
                    caption = self.prefix + f.read()
            else:
                caption = self.prompt

            if random.random() < self.ucg:
                caption = ""
            captions.append(caption)
        return captions

    def get_masks(self, samples):
        masks = torch.stack([torch.tensor(np.load(os.path.join(self.path, "mask", sample + ".npz"))["arr_0"]).unsqueeze(0).repeat(4, 1, 1)
                             for sample in samples])
        masks.to(memory_format=torch.contiguous_format).float()
        return masks

    def get_pfg(self, samples):
        pfg = torch.stack([torch.tensor(np.load(os.path.join(self.path, "pfg", sample + ".npz"))["controll"]).unsqueeze(0)
                           for sample in samples])
        pfg.to(memory_format=torch.contiguous_format).float()
        return pfg
    
    def get_control(self, samples):
        images = []
        for sample in samples:
            image = Image.open(os.path.join(self.path, "control", sample + f".png"))
            image = np.array(image.convert("RGB"))
            image = image[None, :]
            images.append(image)
        images_tensor = np.concatenate(images, axis=0)
        images_tensor = np.array(images_tensor).astype(np.float32) / 255.0
        images_tensor = images_tensor.transpose(0, 3, 1, 2)
        images_tensor = torch.from_numpy(images_tensor).to(memory_format=torch.contiguous_format).float()
        return images_tensor

# controlnetのguide_imageはタスクごとに処理を変えるので継承で対応
class ControlDataset(BaseDataset):   
    def get_control(self, samples):
        images = []
        for sample in samples:
            image = Image.open(os.path.join(self.path[:-4] + "org", sample + f".png"))
            image = np.array(image.convert("RGB"))
            image = image[None, :]
            images.append(image)
        images_tensor = np.concatenate(images, axis=0)
        images_tensor = np.array(images_tensor).astype(np.float32) / 255.0
        images_tensor = images_tensor.transpose(0, 3, 1, 2)
        images_tensor = torch.from_numpy(images_tensor).to(memory_format=torch.contiguous_format).float()
        return images_tensor


class IFDataset(BaseDataset):
    transform = transforms.Compose(
                        [
                        transforms.ToTensor(), 
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])
    def get_latents(self, samples):
        tensors = []
        for sample in samples:
            image = Image.open(os.path.join(self.path[:-4] + "org", sample + f".png"))
            image = expand2square(image, (255,255,255))
            image = image.resize((64,64))
            tensor = self.transform(image)
            tensors.append(tensor)
        tensors = torch.stack(tensors)
        tensors = tensors.to(memory_format=torch.contiguous_format).float()
        return tensors
    
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
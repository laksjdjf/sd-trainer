#mask training refer to https://github.com/cloneofsimo/lora/discussions/96

from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import json
import random
import numpy as np
from transformers import CLIPTokenizer

#dataset:tokenizerとpathと、batch_sizeを引数に取ることが必須。get_itemの出力は"image" or "latents" と"caption"
#SimpleDatasetのbatch sizeはデータローダーの引数で設定する必要がある。

class SimpleDataset(Dataset):
    def __init__(self, tokenizer: CLIPTokenizer, path, batch_size, resolution):
        self.file_list = []
        #画像リスト
        [self.file_list.extend(glob.glob(f'{path}' + '/*.' + e)) for e in ['jpg', 'jpeg', 'png', 'bmp', 'webp']]
        self.transform = transforms.Compose(
        [
        transforms.Resize((resolution,resolution)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ]
    )
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, i):
        img = Image.open(self.file_list[i])
        #txtを読み込む
        with open(os.path.splitext(self.file_list[i])[0] + ".txt" ,"r") as f:
            caption = f.read()
        return {"image":self.transform(img),"caption":caption}

    
#めんどくさいのでlatent cacheとメタデータがあることを前提とする
class AspectDataset(Dataset):
    def __init__(self, tokenizer: CLIPTokenizer, batch_size, path, mask = False, control = False, prompt = None, prefix = ""):
        #メタデータは"(640,896)":["100","101",..]のようなbucketからファイルのリストを出す辞書
        with open(os.path.join(path,"buckets.json"),"r") as f:
            self.bucket2file = json.load(f)
        self.path = path
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        #batchの取り出し方を初期化するメソッド
        self.init_batch_samples()
        
        #マスクをするかどうか
        self.mask = mask
        
        self.controll = control
        
        #全画像共通プロンプト
        self.prompt = prompt
        
        self.prefix = prefix
        
    def __len__(self):
        return len(self.batch_samples) #通常の長さと違ってデータセットの数ではなくミニバッチの数である。
    
    def __getitem__(self, i):
        if i == 0:
            self.init_batch_samples()
            
        samples = self.batch_samples[i]
        #latentを読み込む
        latents = torch.stack([torch.tensor(np.load(os.path.join(self.path, sample + ".npy"))) for sample in samples])
        #なんかみんなやってるからやるけどこれなーんだ？
        latents = latents.to(memory_format=torch.contiguous_format).float()
        
        #captionを読み込む
        captions = []
        for sample in samples:
            if self.prompt is None:
                with open(os.path.join(self.path,sample + ".caption" ),"r") as f:
                    captions.append(self.prefix + f.read())
            else:    
                captions = [self.prompt for _ in range(len(samples))]
        
        
        if self.mask:
            masks = torch.stack([torch.tensor(np.load(os.path.join(self.path, sample + ".npz"))["arr_0"]).unsqueeze(0).repeat(4,1,1) for sample in samples])
            masks.to(memory_format=torch.contiguous_format).float()
        else:
            masks = None
        
        if self.controll:
            controls = torch.stack([torch.tensor(np.load(os.path.join(self.path, sample + ".npz"))["controll"]).unsqueeze(0) for sample in samples])
            controls.to(memory_format=torch.contiguous_format).float()
        else:
            controls = None
            
        return {"latents":latents,"caption":captions,"mask": masks,"control": controls}
                  
    def init_batch_samples(self):
        random.seed()
        self.batch_samples = []
        for key in  self.bucket2file:
            #bucketごとにファイルの順番をシャッフル
            random.shuffle(self.bucket2file[key])
            #batch sizeごとにファイルを取り出す
            self.batch_samples.extend([self.bucket2file[key][i:i+self.batch_size] for i in range(0,len(self.bucket2file[key]),self.batch_size)])
        #できたリストもシャッフル
        random.shuffle(self.batch_samples)
        

#めんどくさいのでlatent cacheとメタデータがあることを前提とする
class FIDDataset(Dataset):
    def __init__(self, tokenizer: CLIPTokenizer, batch_size, path, mask = False, control = False, prompt = None):
        #メタデータは"(640,896)":["100","101",..]のようなbucketからファイルのリストを出す辞書
        with open(os.path.join(path,"buckets.json"),"r") as f:
            self.bucket2file = json.load(f)
        self.path = path
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        #batchの取り出し方を初期化するメソッド
        self.init_batch_samples()
        
        #マスクをするかどうか
        self.mask = mask
        
        self.controll = control
        
        #全画像共通プロンプト
        self.prompt = prompt
        
    def __len__(self):
        return len(self.batch_samples) #通常の長さと違ってデータセットの数ではなくミニバッチの数である。
    
    def __getitem__(self, i):
        if i == 0:
            self.init_batch_samples()
            
        samples = self.batch_samples[i]
        
        #captionを読み込む
        captions = []
        for sample in samples:
            if self.prompt is None:
                with open(os.path.join(self.path,sample + ".caption" ),"r") as f:
                    captions.append(f.read())
            else:    
                captions = [self.prompt for _ in range(len(samples))]
        
        if self.controll:
            controls = torch.stack([torch.tensor(np.load(os.path.join(self.path, sample + ".npz"))["controll"]).unsqueeze(0) for sample in samples])
            controls.to(memory_format=torch.contiguous_format).float()
        else:
            controls = None
            
        return {"caption":captions,"control": controls}
                  
    def init_batch_samples(self):
        self.batch_samples = []
        for key in  self.bucket2file:
            #bucketごとにファイルの順番をシャッフル
            random.shuffle(self.bucket2file[key])
            #batch sizeごとにファイルを取り出す
            self.batch_samples.extend([self.bucket2file[key][i:i+self.batch_size] for i in range(0,len(self.bucket2file[key]),self.batch_size)])
        #できたリストもシャッフル
        random.shuffle(self.batch_samples)
        

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

class SimpleDataset(Dataset):
    def __init__(self, path, size):
        self.file_list = []
        #画像リスト
        [self.file_list.extend(glob.glob(f'{path}' + '/*.' + e)) for e in ['jpg', 'jpeg', 'png', 'bmp', 'webp']]
        self.transform = transforms.Compose(
        [
        transforms.Resize(size),
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
    def __init__(self, path, tokenizer: CLIPTokenizer, batch_size = 1):
        #メタデータは"(640,896)":["100","101",..]のようなbucketからファイルのリストを出す辞書
        with open(os.path.join(path,"buckets.json"),"r") as f:
            self.bucket2file = json.load(f)
        self.path = path
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        #batchの取り出し方を初期化するメソッド
        self.init_batch_samples()
        
    def __len__(self):
        return len(self.batch_samples) #通常の長さと違ってデータセットの数ではなくミニバッチの数である。
    
    def __getitem__(self, i):
        if i == 0:
            self.init_batch_samples()
            
        samples = self.batch_samples[i]
        #latentを読み込む
        latents = torch.stack([torch.tensor(np.load(os.path.join(self.path, sample + ".npy"))) for sample in samples])
        
        #captionを読み込む
        captions = []
        for sample in samples:
            with open(os.path.join(self.path,sample + ".caption" ),"r") as f:
                captions.append(f.read())
        tokens = self.tokenizer(captions, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors='pt').input_ids
        return {"latents":latents,"tokens":tokens}
                  
    def init_batch_samples(self):
        self.batch_samples = []
        for key in  self.bucket2file:
            #bucketごとにファイルの順番をシャッフル
            random.shuffle(self.bucket2file[key])
            #batch sizeごとにファイルを取り出す
            self.batch_samples.extend([self.bucket2file[key][i:i+self.batch_size] for i in range(0,len(self.bucket2file[key]),self.batch_size)])
        #できたリストもシャッフル
        random.shuffle(self.batch_samples)
        
        
        
        

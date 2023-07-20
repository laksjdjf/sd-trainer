# original dataset for sd-scripts

from library.train_util import MinimalDataset
import json
import random
import torch
import os
import numpy as np

class OriginalDataset(MinimalDataset):
    def __init__(self, tokenizer, max_token_length, resolution, debug_dataset=False):
        super().__init__(tokenizer, max_token_length, resolution, debug_dataset)
        self.batch_size = 8
        self.path = "dataset"
        with open(os.path.join(self.path, "buckets.json"), "r") as f:
            self.bucket2file = json.load(f)
        self.init_batch_samples()

    def __len__(self):
        return len(self.batch_samples)

    def __getitem__(self, i):
        batch = {}
        samples = self.batch_samples[i]

        latents = self.get_latents(samples)
        batch["latents"] = latents
        captions = self.get_captions(samples)
        batch["captions"] = captions
        
        batch["input_ids"] = torch.stack([self.get_input_ids(caption, self.tokenizer[0]) for caption in captions])
        if len(self.tokenizer) > 1:
            batch["input_ids2"] = torch.stack([self.get_input_ids(caption, self.tokenizer[1]) for caption in captions])
        
        batch["loss_weights"] = torch.ones(latents.shape[0], dtype=torch.float32)

        size = [latents.shape[2] * 8, latents.shape[3] * 8]
        batch["original_sizes_hw"] = torch.stack([torch.LongTensor(size)]*latents.shape[0])
        batch["crop_top_lefts"] = torch.stack([torch.LongTensor([0,0])]*latents.shape[0])
        batch["target_sizes_hw"] = torch.stack([torch.LongTensor(size)]*latents.shape[0])

        return batch
    
    def get_latents(self, samples):
        latents = torch.stack([torch.tensor(np.load(os.path.join(self.path, "latents", sample + ".npy"))) for sample in samples])
        latents = latents.to(memory_format=torch.contiguous_format).float()  # これなに
        return latents

    def get_captions(self, samples):
        captions = []
        for sample in samples:
            with open(os.path.join(self.path, "captions", sample + ".caption"), "r") as f:
                caption = f.read()
            captions.append(caption)
            
        return captions
    
    def init_batch_samples(self):
        self.batch_samples = []
        for key in self.bucket2file:
            random.shuffle(self.bucket2file[key])
            self.batch_samples.extend([self.bucket2file[key][i:i+self.batch_size]
                                      for i in range(0, len(self.bucket2file[key]), self.batch_size)])
        random.shuffle(self.batch_samples)

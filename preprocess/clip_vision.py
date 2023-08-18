import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import glob
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from utils import preprocess_for_clip_vision

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--directory', '-d', type=str, required=True, help='入力画像のディレクトリ')
parser.add_argument('--output_path', '-o', type=str, required=True, help='出力先ディレクトリ')
parser.add_argument('--clip_path', '-p', type=str, default="stabilityai/stable-diffusion-2-1-unclip", required=False, help='clipのパス')
parser.add_argument('--start', '-s', required=False, default=0, type=int)
parser.add_argument('--end', '-e', required=False, type=int)
parser.add_argument('--batch_size', '-b', required=False, default=64, type=int)
args = parser.parse_args()

class CLIPVisionDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list
        self.clip_image_processor = CLIPImageProcessor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = Image.open(path)
        image = preprocess_for_clip_vision(image)
        image = self.clip_image_processor(images=image, return_tensors="pt").pixel_values[0]
        file_name = os.path.basename(path)
        file_name = os.path.splitext(file_name)[0]
        return {"image":image, "file_name":file_name}

#WD 1.4 tagger
@torch.no_grad()
def main():
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.clip_path, subfolder="image_encoder").to("cuda", dtype=torch.float16)
    
    path = args.directory
    output_path = args.output_path
    
    file_list = []
    [file_list.extend(glob.glob(f'{path}' + '/*.' + e)) for e in ['jpg', 'jpeg', 'png', 'bmp', 'webp']]
    
    start_id = args.start
    end_id = args.end if args.end is not None else len(file_list)
    
    file_list = file_list[start_id:end_id]

    file_list = [os.path.join(path, file) for file in file_list]

    dataset = CLIPVisionDataset(file_list)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    os.makedirs(output_path, exist_ok=True)

    for batch in tqdm(dataloader):
        images = batch["image"].to("cuda", dtype=torch.float16)
        file_names = batch["file_name"]

        embeddings = image_encoder(images).image_embeds
        embeddings = embeddings.float().cpu().numpy()
        for j in range(len(file_names)):
            np.save(os.path.join(output_path, file_names[j] + ".npy"), embeddings[j])

if __name__ == "__main__":
    main()

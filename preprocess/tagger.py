import os

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import glob
from torch.utils.data import Dataset, DataLoader

from model import ViT
from utils import preprocess_for_tagger

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--directory', '-d', type=str, required=True, help='入力画像のディレクトリ')
parser.add_argument('--output_path', '-o', type=str, required=True, help='出力先ディレクトリ')
parser.add_argument('--tagger_path', '-p', type=str, default="wd-v1-4-tagger-pytorch", required=False, help='taggerのパス')
parser.add_argument('--start', '-s', required=False, default=0, type=int)
parser.add_argument('--end', '-e', required=False, type=int)
parser.add_argument('--image_size', '-i', required=False, default=448, type=int)
parser.add_argument('--batch_size', '-b', required=False, default=64, type=int)
parser.add_argument('--threshold', '-t', required=False, default=0.35, type=float, help='閾値')
parser.add_argument('--make_caption', '-m', action='store_true', help='taggerの出力を直接captionにする')
args = parser.parse_args()

class TaggerDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = Image.open(path)
        image = preprocess_for_tagger(image)
        img = torch.tensor(image).permute(2,0,1)
        file_name = os.path.basename(path)
        file_name = os.path.splitext(file_name)[0]
        return {"image":img, "file_name":file_name}

#WD 1.4 tagger
@torch.no_grad()
def main():
    model = ViT(3, 448, 9083)
    model.load_state_dict(torch.load(os.path.join(args.tagger_path,"wd-v1-4-vit-tagger-v2.ckpt")))
    model.eval()
    model.cuda()
    
    label_names = pd.read_csv(os.path.join(args.tagger_path,"selected_tags.csv"))
    
    path = args.directory
    output_path = args.output_path
    
    file_list = []
    [file_list.extend(glob.glob(f'{path}' + '/*.' + e)) for e in ['jpg', 'jpeg', 'png', 'bmp', 'webp']]
    
    start_id = args.start
    end_id = args.end if args.end is not None else len(file_list)
    
    file_list = file_list[start_id:end_id]

    file_list = [os.path.join(path, file) for file in file_list]

    dataset = TaggerDataset(file_list)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    os.makedirs(output_path, exist_ok=True)

    for batch in tqdm(dataloader):
        images = batch["image"].cuda()
        file_names = batch["file_name"]

        probs = model(images)
        probs = probs.cpu().numpy()

        for j in range(len(file_names)):
            label_names["probs"] = probs[j]
            tags_names = label_names[label_names["category"] == 0]
            found_tags = tags_names[tags_names["probs"] > args.threshold].sort_values("probs",ascending=False)[["name"]]
            tags = " ".join(list(found_tags["name"]))
            if args.make_caption:
                with open(os.path.join(output_path, file_names[j] + ".caption"),"w") as f:
                    f.write(tags)
            else:
                with open(os.path.join(path, file_names[j] + ".txt"),"r") as f:
                    caption = f.read()
                with open(os.path.join(output_path, file_names[j] + ".caption"),"w") as f:
                    f.write(caption[:-1] + ', "tagger": "' + tags + '"}')

if __name__ == "__main__":
    main()

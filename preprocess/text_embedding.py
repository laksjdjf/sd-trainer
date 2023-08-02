from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn as nn

import os
import numpy as np
import sys
sys.path.insert(0, './') # あまり賢くないが、sd-trainer直下で実行する
from utils.model import TextModel

args = ArgumentParser()
args.add_argument('--model_path', '-m', type=str, required=True)
args.add_argument('--dataset_dir', '-d', type=str, required=True)
args.add_argument('--output_dir' , '-o', type=str, required=True)
args.add_argument('--start', '-s', type=int, default=0)
args.add_argument('--end', '-e', type=int, default=None)
args.add_argument('--batch_size', '-b', type=int, default=32)

@torch.no_grad()
def main(args):
    model = TextModel.from_pretrained(args.model_path, sdxl=True, clip_skip=-2)
    model.eval()
    model.to("cuda")
    model.requires_grad_(False)
    
    files = [file for file in os.listdir(args.dataset_dir) if "caption" in file]
    end = args.end or len(files)
    files = files[args.start:args.end]
    os.makedirs(args.output_dir, exist_ok=True)

    for i in tqdm(range(0,len(files),args.batch_size)):
        # text fileを読み込む
        batch_files = files[i:i+args.batch_size]
        batch_texts = []
        for file in batch_files:
            with open(os.path.join(args.dataset_dir, file), "r") as f:
                batch_texts.append(f.read())
        tokens, tokens_2, empty_text = model.tokenize(batch_texts)
        encoder_hidden_states, pooled_outputs = model(tokens, tokens_2, empty_text)
        encoder_hidden_states = encoder_hidden_states.cpu().numpy()
        pooled_outputs = pooled_outputs.cpu().numpy()
        for file, encoder_hidden_state, pooled_output in zip(batch_files, encoder_hidden_states, pooled_outputs):
            output_path = os.path.join(args.output_dir,os.path.splitext(file)[0] + ".npz")
            np.savez(output_path, encoder_hidden_state=encoder_hidden_state, pooled_output=pooled_output)


if __name__ == "__main__":
    args = args.parse_args()
    main(args)


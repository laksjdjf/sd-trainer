from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn as nn

import os
import numpy as np
import sys
sys.path.insert(0, './') # あまり賢くないが、sd-trainer直下で実行する
from modules.text_model import SD1TextModel, SDXLTextModel, FluxTextModel

args = ArgumentParser()
args.add_argument('--model_path', '-m', type=str, required=True)
args.add_argument('--model_type', '-t', type=str, default="sdxl") # sdxl, sd1, flux
args.add_argument('--dataset_dir', '-d', type=str, required=True)
args.add_argument('--output_dir' , '-o', type=str, required=True)
args.add_argument('--start', '-s', type=int, default=0)
args.add_argument('--end', '-e', type=int, default=None)
args.add_argument('--batch_size', '-b', type=int, default=32)
args.add_argument('--dtype', '-dt', type=str, default="bfloat16", choices=["float16", "float32", "bfloat16"])

@torch.no_grad()
def main(args):
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise ValueError("dtype must be float16, float32, or bfloat16")
    if args.model_type == "sdxl":
        model = SDXLTextModel.from_pretrained(args.model_path, clip_skip=-2, torch_dtype=dtype)
    elif args.model_type == "sd1":
        model = SD1TextModel.from_pretrained(args.model_path, clip_skip=-2, torch_dtype=dtype)
    elif args.model_type == "flux":
        model = FluxTextModel.from_pretrained(args.model_path, clip_skip=-2, torch_dtype=dtype)
    else:
        raise ValueError("model_type must be sdxl, sd1, or flux")
    
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
        text_output = model(batch_texts)
        encoder_hidden_states, pooled_outputs = text_output.encoder_hidden_states, text_output.pooled_output
        encoder_hidden_states = encoder_hidden_states.cpu().numpy()
        pooled_outputs = pooled_outputs.cpu().numpy()
        for file, encoder_hidden_state, pooled_output in zip(batch_files, encoder_hidden_states, pooled_outputs):
            output_path = os.path.join(args.output_dir,os.path.splitext(file)[0] + ".npz")
            np.savez(output_path, encoder_hidden_state=encoder_hidden_state, pooled_output=pooled_output)

if __name__ == "__main__":
    args = args.parse_args()
    main(args)


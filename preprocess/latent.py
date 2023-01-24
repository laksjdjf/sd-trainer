import os 
import numpy as np
import torch
from torchvision import transforms
from diffusers import AutoencoderKL
from tqdm import tqdm
import argparse
from PIL import Image
import json


parser = argparse.ArgumentParser()
parser.add_argument('--directory', '-d', type=str, required=True)
parser.add_argument('--output_path', '-o', type=str, required=True)
parser.add_argument('--start', '-s', required=False, default=0, type=int)
parser.add_argument('--end', '-e', required=False, type=int)
parser.add_argument('--model','-m' , required=True, type=str)
parser.add_argument('--batch_size','-b' , required=False, default=10, type=int)
args = parser.parse_args()

def main():
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae")
    vae.eval()
    vae.to("cuda", dtype=torch.float16)
    
    to_tensor_norm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    path = args.directory.rstrip("/") + "/"
    output_path = args.output_path.rstrip("/") + "/"
    
    files = os.listdir(args.directory)
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    end_id = args.end if args.end is not None else len(files)
    
    for file in tqdm(files[args.start:end_id]):
        if "png" not in file:
            continue
        image = Image.open(path + file)
        image_tensor = to_tensor_norm(image).to("cuda",torch.float16)
        image_tensors = torch.stack([image_tensor]) #batch size 1のごみ実装
        with torch.no_grad():
            latent = vae.encode(image_tensors).latent_dist.sample().float().to("cpu").numpy()[0]
        np.save(output_path + file[:-4] + ".npy",latent)
        
def main_batch():
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae")
    vae.eval()
    vae.to("cuda", dtype=torch.float16)
    
    to_tensor_norm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    path = args.directory.rstrip("/") + "/"
    output_path = args.output_path.rstrip("/") + "/"
    
    exist_files = [file[:-4] for file in os.listdir(output_path) if ".npy" in file]
    
    with open(path + "buckets.json" ,"r") as f:
        buckets = json.load(f)
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    
    for key in buckets.keys():
        image_tensors = []
        files = []
        print(key)
        for file in tqdm(buckets[key]):
            if file in exist_files:
                continue
            image = Image.open(path + file + ".png")
            image_tensor = to_tensor_norm(image).to("cuda",torch.float16)
            image_tensors.append(image_tensor)
            files.append(file)
            if len(files) == args.batch_size:
                input_tensor = torch.stack(image_tensors)
                with torch.no_grad():
                    latents = vae.encode(input_tensor).latent_dist.sample().float().to("cpu").numpy()
                for i in range(len(files)):
                    np.save(output_path + files[i] + ".npy",latents[i])
                image_tensors = []
                files = []
        
        if len(files) > 0:
            input_tensor = torch.stack(image_tensors)
            with torch.no_grad():
                latents = vae.encode(input_tensor).latent_dist.sample().float().to("cpu").numpy()
            for i in range(len(files)):
                np.save(output_path + files[i] + ".npy",latents[i])
                
def main_same_size():
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae")
    vae.eval()
    vae.to("cuda", dtype=torch.float16)
    
    to_tensor_norm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    path = args.directory.rstrip("/") + "/"
    output_path = args.output_path.rstrip("/") + "/"
    
    all_files = os.listdir(args.directory)
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    
    image_tensors = []
    files = []
    for file in tqdm(all_files):
        if "txt" in file:
            continue
        image = Image.open(path + file)
        image_tensor = to_tensor_norm(image).to("cuda",torch.float16)
        image_tensors.append(image_tensor)
        files.append(file)
        if len(files) == args.batch_size:
            input_tensor = torch.stack(image_tensors)
            with torch.no_grad():
                latents = vae.encode(input_tensor).latent_dist.sample().float().to("cpu").numpy()
            for i in range(len(files)):
                np.save(output_path + files[i][:-4] + ".npy",latents[i])
            image_tensors = []
            files = []

    if len(files) > 0:
        input_tensor = torch.stack(image_tensors)
        with torch.no_grad():
            latents = vae.encode(input_tensor).latent_dist.sample().float().to("cpu").numpy()
        for i in range(len(files)):
            np.save(output_path + files[i][:-4] + ".npy",latents[i])
    
                
if __name__ == "__main__":
    main_batch()

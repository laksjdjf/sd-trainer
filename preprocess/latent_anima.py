import argparse
import json
import os

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--directory", "-d", type=str, required=True)
parser.add_argument("--output_path", "-o", type=str, required=True)
parser.add_argument("--model", "-m", type=str, required=True)
parser.add_argument("--batch_size", "-b", type=int, default=1)
parser.add_argument("--dtype", "-t", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
parser.add_argument("--metadata", type=str, default="buckets.json")
args = parser.parse_args()


def get_dtype(name):
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError("dtype must be fp32, fp16, or bf16")


def check_and_assert_nan_tensor(tensor):
    if torch.isnan(tensor).any().item():
        raise ValueError("nan tensor")


def load_vae(model, dtype):
    if os.path.isfile(model):
        raise ValueError(
            "Official Diffusers Anima support requires a Diffusers-format model directory or Hub repository; "
            "single-file checkpoints are not supported."
        )
    vae = AutoencoderKLQwenImage.from_pretrained(model, subfolder="vae", torch_dtype=dtype)
    vae.eval()
    vae.to("cuda", dtype=dtype)
    return vae


def encode_batch(vae, image_tensors):
    input_tensor = torch.stack(image_tensors).unsqueeze(2)
    with torch.no_grad():
        latents = vae.encode(input_tensor).latent_dist.sample()
        check_and_assert_nan_tensor(latents)
    return latents.float().cpu().numpy()


def main():
    dtype = get_dtype(args.dtype)
    vae = load_vae(args.model, dtype)

    to_tensor_norm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    path = args.directory.rstrip("/") + "/"
    output_path = args.output_path.rstrip("/") + "/"
    os.makedirs(output_path, exist_ok=True)

    exist_files = {file[:-4] for file in os.listdir(output_path) if file.endswith(".npy")}
    with open(os.path.join(path, args.metadata), "r") as f:
        buckets = json.load(f)

    for key, samples in buckets.items():
        print(key)
        image_tensors = []
        files = []
        for sample in tqdm(samples):
            if sample in exist_files:
                continue
            image = Image.open(os.path.join(path, sample + ".png")).convert("RGB")
            image_tensors.append(to_tensor_norm(image).to("cuda", dtype=dtype))
            files.append(sample)
            if len(files) == args.batch_size:
                latents = encode_batch(vae, image_tensors)
                for i, file in enumerate(files):
                    np.save(os.path.join(output_path, file + ".npy"), latents[i])
                image_tensors = []
                files = []

        if files:
            latents = encode_batch(vae, image_tensors)
            for i, file in enumerate(files):
                np.save(os.path.join(output_path, file + ".npy"), latents[i])


if __name__ == "__main__":
    with torch.no_grad():
        main()

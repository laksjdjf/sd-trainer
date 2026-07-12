# WD Taggerによるタグ付け(caption生成)。timm経由でHuggingFace Hubからロードする。
import os

import numpy as np
import pandas as pd
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader, Dataset

from preprocess.common import list_images, progress_iter

# 「0_0」みたいな顔文字タグはアンダースコアを残す
KAOMOJIS = [
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=",
    ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
]


def pad_to_square(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(background, image).convert("RGB")
    w, h = image.size
    size = max(w, h)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(image, ((size - w) // 2, (size - h) // 2))
    return canvas


class TaggerDataset(Dataset):
    def __init__(self, file_list, transform):
        self.files = file_list
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = pad_to_square(Image.open(path))
        tensor = self.transform(image)
        tensor = tensor.flip(0)  # RGB -> BGR
        file_name = os.path.splitext(os.path.basename(path))[0]
        return {"image": tensor, "file_name": file_name}


def format_tag(name: str) -> str:
    if name in KAOMOJIS:
        return name
    return name.replace("_", " ")


@torch.no_grad()
def tag_images(
    directory,
    output_dir,
    caption_dir=None,
    repo_id="SmilingWolf/wd-eva02-large-tagger-v3",
    batch_size=16,
    threshold=0.35,
    character_threshold=0.75,
    extension="caption",
    num_workers=4,
    on_progress=None,
):
    model = timm.create_model(f"hf-hub:{repo_id}", pretrained=True)
    model.eval().cuda()
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    labels = pd.read_csv(hf_hub_download(repo_id, "selected_tags.csv"))
    general_mask = (labels["category"] == 0).to_numpy()
    character_mask = (labels["category"] == 4).to_numpy()
    names = labels["name"].to_numpy()

    file_list = list_images(directory)
    dataset = TaggerDataset(file_list, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    os.makedirs(output_dir, exist_ok=True)

    for batch in progress_iter(dataloader, total=len(dataloader), desc="tagging", on_progress=on_progress):
        images = batch["image"].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probs = torch.sigmoid(model(images)).float().cpu().numpy()

        for file_name, prob in zip(batch["file_name"], probs):
            general = (prob > threshold) & general_mask
            character = (prob > character_threshold) & character_mask
            selected = general | character
            order = np.argsort(prob[selected])[::-1]
            tags = ", ".join(format_tag(name) for name in names[selected][order])

            if caption_dir is not None:
                with open(os.path.join(caption_dir, file_name + ".caption")) as f:
                    caption = f.read().strip()
                text = f"{tags}, {caption}" if caption else tags
            else:
                text = tags

            with open(os.path.join(output_dir, f"{file_name}.{extension}"), "w") as f:
                f.write(text)

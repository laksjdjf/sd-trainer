# VAEによるlatentキャッシュの作成。モデルタイプごとのVAEクラスはmodel_registryから解決する。
import json
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from modules.model_registry import get_model_spec
from preprocess.common import assert_not_nan, list_images, parse_dtype, progress_iter


def load_vae(model_type, model_path, dtype, revision=None):
    spec = get_model_spec(model_type)
    if spec.vae_cls is None:
        raise ValueError(f"{model_type}のVAE単体ロードは未対応だよ。")
    if os.path.isfile(model_path):
        raise ValueError("latentキャッシュはDiffusers形式のモデルディレクトリかHubリポジトリにしか対応してないよ。")
    vae = spec.vae_cls.from_pretrained(model_path, subfolder="vae", revision=revision, torch_dtype=dtype)
    vae.eval()
    vae.to("cuda", dtype=dtype)
    return vae


@torch.no_grad()
def _encode_batch(vae, image_tensors, video):
    x = torch.stack(image_tensors)
    if video:  # animaのようなvideoモデルは時間次元を付与する
        x = x.unsqueeze(2)
    latents = vae.encode(x).latent_dist.sample()
    assert_not_nan(latents)
    return latents.float().cpu().numpy()


def _iter_batches(directory, metadata, batch_size):
    # buckets.jsonがあればbucketごとにbatch_size件ずつ、なければサイズ不明なので1件ずつ
    metadata_path = os.path.join(directory, metadata)
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            buckets = json.load(f)
        for samples in buckets.values():
            for i in range(0, len(samples), batch_size):
                yield samples[i:i + batch_size]
    else:
        for file in list_images(directory):
            yield [os.path.splitext(os.path.basename(file))[0]]


@torch.no_grad()
def encode_latents(
    directory,
    output_dir,
    model_path,
    model_type="sdxl",
    dtype="fp32",
    batch_size=8,
    metadata="buckets.json",
    skip_existing=True,
    revision=None,
    on_progress=None,
):
    dtype = parse_dtype(dtype)
    vae = load_vae(model_type, model_path, dtype, revision)
    video = model_type == "anima"

    to_tensor_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    os.makedirs(output_dir, exist_ok=True)
    exist_files = {os.path.splitext(f)[0] for f in os.listdir(output_dir) if f.endswith(".npy")} if skip_existing else set()

    batches = [
        [s for s in batch if s not in exist_files]
        for batch in _iter_batches(directory, metadata, batch_size)
    ]
    batches = [batch for batch in batches if batch]
    total = sum(len(batch) for batch in batches)

    for batch in progress_iter(batches, total=len(batches), desc=f"latents ({total} files)", on_progress=on_progress):
        image_tensors = [
            to_tensor_norm(Image.open(os.path.join(directory, sample + ".png")).convert("RGB")).to("cuda", dtype=dtype)
            for sample in batch
        ]
        latents = _encode_batch(vae, image_tensors, video)
        for i, sample in enumerate(batch):
            np.save(os.path.join(output_dir, sample + ".npy"), latents[i])

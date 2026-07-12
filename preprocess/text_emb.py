# テキストエンコーダ出力(text embedding)のキャッシュ作成。モデルクラスはmodel_registryから解決する。
import os

import numpy as np
import torch

from modules.model_registry import get_model_spec
from preprocess.common import parse_dtype, progress_iter


def load_text_model(model_type, model_path, dtype, clip_skip=None, revision=None):
    spec = get_model_spec(model_type)
    if spec.text_model_cls is None:
        raise ValueError(f"{model_type}のテキストモデル単体ロードは未対応だよ。")
    kwargs = {}
    if "clip_skip" in spec.text_extra_args:
        kwargs["clip_skip"] = clip_skip if clip_skip is not None else spec.default_clip_skip
    model = spec.text_model_cls.from_pretrained(model_path, revision=revision, torch_dtype=dtype, **kwargs)
    model.eval()
    model.to("cuda")
    model.requires_grad_(False)
    return model


@torch.no_grad()
def encode_text(
    dataset_dir,
    output_dir,
    model_path,
    model_type="sdxl",
    dtype="bf16",
    batch_size=32,
    clip_skip=None,
    save_dtype="fp16",
    revision=None,
    on_progress=None,
):
    dtype = parse_dtype(dtype)
    save_dtype = "float16" if save_dtype in ("fp16", "float16") else "float32"
    model = load_text_model(model_type, model_path, dtype, clip_skip, revision)

    files = sorted(f for f in os.listdir(dataset_dir) if f.endswith(".caption"))
    os.makedirs(output_dir, exist_ok=True)

    batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
    for batch_files in progress_iter(batches, total=len(batches), desc="text_emb", on_progress=on_progress):
        batch_texts = []
        for file in batch_files:
            with open(os.path.join(dataset_dir, file), "r") as f:
                batch_texts.append(f.read())

        text_output = model(batch_texts)
        encoder_hidden_states = text_output.encoder_hidden_states.float().cpu().numpy().astype(save_dtype)
        pooled_outputs = text_output.pooled_output.float().cpu().numpy().astype(save_dtype)

        for file, encoder_hidden_state, pooled_output in zip(batch_files, encoder_hidden_states, pooled_outputs):
            output_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".npz")
            np.savez(output_path, encoder_hidden_state=encoder_hidden_state, pooled_output=pooled_output)

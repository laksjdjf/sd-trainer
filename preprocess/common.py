# preprocess共通のユーティリティ。
import glob
import os

from tqdm import tqdm

IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "bmp", "webp")

# fp16/float16のような略記と正式名の両方を受け付ける
DTYPE_ALIASES = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
}


def parse_dtype(dtype):
    # torch.dtypeそのものが来たらそのまま返す
    import torch
    if isinstance(dtype, torch.dtype):
        return dtype
    name = DTYPE_ALIASES.get(dtype, dtype)
    if name not in ("float32", "float16", "bfloat16"):
        raise ValueError(f"dtype={dtype}は未対応。fp32/fp16/bf16のどれかにしてね。")
    return getattr(torch, name)


def list_images(directory):
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(directory, f"*.{ext}")))
    return sorted(files)


def progress_iter(iterable, total=None, desc=None, on_progress=None):
    # on_progressがNoneならtqdmで表示し、指定されていればコールバックへ進捗を渡す(UI連携用)
    if on_progress is None:
        yield from tqdm(iterable, total=total, desc=desc)
        return
    count = 0
    for item in iterable:
        count += 1
        on_progress(count, total, desc)
        yield item


def assert_not_nan(tensor):
    import torch
    if torch.isnan(tensor).any().item():
        raise ValueError("nan tensor")

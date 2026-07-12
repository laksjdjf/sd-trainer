# アスペクト比bucketの生成と、画像のbucketへの振り分け(リサイズ+センタークロップ)。
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from PIL import Image

from preprocess.common import list_images, progress_iter


def make_buckets(resolution=1024, min_length=512, max_length=2048, max_ratio=2.0):
    # モデルの構造からして64の倍数が推奨される。(VAEで8分の1⇒UNetで8分の1)
    increment = 64
    max_pixels = resolution * resolution

    # 正方形は手動で追加
    buckets = {(resolution, resolution)}

    width = min_length
    while width <= max_length:
        # 最大ピクセル数と最大長を越えない最大の高さ
        height = min(max_length, (max_pixels // width) - (max_pixels // width) % increment)
        ratio = width / height
        # アスペクト比が極端じゃなかったら追加、高さと幅を入れ替えたものも追加
        if 1 / max_ratio <= ratio <= max_ratio:
            buckets.add((width, height))
            buckets.add((height, width))
        width += increment

    buckets = list(buckets)
    ratios = [w / h for w, h in buckets]
    buckets = np.array(buckets)[np.argsort(ratios)]
    ratios = np.sort(ratios)
    return buckets, ratios


def _get_png_resolution(filepath):
    with open(filepath, "rb") as f:
        header = f.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Not a PNG file")
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        return width, height


def _resize_to_bucket(file, output_dir, buckets, ratios):
    # 出力先に既にあればbucket判定のみ、なければ一番近いbucketへリサイズ+センタークロップして保存
    filepath = os.path.join(output_dir, os.path.splitext(os.path.basename(file))[0] + ".png")
    exists = os.path.exists(filepath)
    if exists:
        width, height = _get_png_resolution(filepath)
    else:
        image = Image.open(file).convert("RGB")
        width, height = image.size
    ratio = width / height

    ar_errors = ratios - ratio
    indice = np.argmin(np.abs(ar_errors))  # 一番近いアスペクト比のインデックス
    bucket_width, bucket_height = (int(x) for x in buckets[indice])  # np.int64のままだとjsonキーが汚れる
    ar_error = ar_errors[indice]

    if not exists:
        if ar_error <= 0:  # bucketより縦長なら高さを合わせて左右を切る
            temp_width = int(width * bucket_height / height)
            image = image.resize((temp_width, bucket_height))
            left = (temp_width - bucket_width) / 2
            image = image.crop((left, 0, bucket_width + left, bucket_height))
        else:  # bucketより横長なら幅を合わせて上下を切る
            temp_height = int(height * bucket_width / width)
            image = image.resize((bucket_width, temp_height))
            upper = (temp_height - bucket_height) / 2
            image = image.crop((0, upper, bucket_width, bucket_height + upper))
        image.save(filepath)

    return os.path.splitext(os.path.basename(file))[0], str((bucket_width, bucket_height))


def bucket_images(
    input_dir,
    output_dir,
    resolution=1024,
    min_length=512,
    max_length=2048,
    max_ratio=2.0,
    num_workers=12,
    on_progress=None,
):
    # 画像をbucketへ振り分けてoutput_dirに保存し、buckets.jsonを書き出す
    os.makedirs(output_dir, exist_ok=True)
    buckets, ratios = make_buckets(resolution, min_length, max_length, max_ratio)
    files = list_images(input_dir)

    worker = partial(_resize_to_bucket, output_dir=output_dir, buckets=buckets, ratios=ratios)
    with ProcessPoolExecutor(num_workers) as e:
        results = list(progress_iter(e.map(worker, files), total=len(files), desc="bucketing", on_progress=on_progress))

    meta = {}
    for file, bucket in results:
        meta.setdefault(bucket, []).append(file)

    with open(os.path.join(output_dir, "buckets.json"), "w") as f:
        json.dump(meta, f)
    return meta


def _image_bucket(file, directory):
    try:
        with Image.open(os.path.join(directory, file)) as img:
            bucket = str((img.width, img.height))
    except Exception:
        return None
    return os.path.splitext(file)[0], bucket


def make_metadata(directory, num_workers=8, on_progress=None):
    # リサイズ済みデータセットからbuckets.jsonだけを作り直す
    files = [os.path.basename(f) for f in list_images(directory)]

    worker = partial(_image_bucket, directory=directory)
    with ProcessPoolExecutor(num_workers) as e:
        results = list(progress_iter(e.map(worker, files, chunksize=64), total=len(files), desc="metadata", on_progress=on_progress))

    meta = {}
    for result in results:
        if result is None:
            continue
        file, bucket = result
        meta.setdefault(bucket, []).append(file)

    with open(os.path.join(directory, "buckets.json"), "w") as f:
        json.dump(meta, f)
    return meta


def make_original_size_metadata(metadata_path, output_path):
    # スクレイピング時のメタデータ(image_width/image_height)からSDXLのsize condition用jsonを作る
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    dic = {}
    for key, value in meta.items():
        if "image_width" in value:
            dic[key] = {
                "original_width": int(value["image_width"]),
                "original_height": int(value["image_height"]),
            }

    with open(output_path, "w") as f:
        json.dump(dic, f)
    return dic

# アニメ顔検出による学習用マスクの作成。lbpcascade_animeface.xmlが必要。
# https://github.com/nagadomi/lbpcascade_animeface
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from preprocess.common import list_images, progress_iter

_classifier = None  # プロセスごとに1回だけロードする


def _detect(file, directory, output_dir, cascade_path):
    import cv2
    global _classifier
    if _classifier is None:
        _classifier = cv2.CascadeClassifier(cascade_path)

    image = cv2.imread(os.path.join(directory, os.path.basename(file)))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _classifier.detectMultiScale(gray_image)

    # latentのサイズは画像サイズ÷8。顔部分以外も完全に0にはしない
    mask = np.full((image.shape[0] // 8, image.shape[1] // 8), 0.05)
    for x, y, w, h in faces:
        mask[y // 8:(y + h) // 8, x // 8:(x + w) // 8] = 1

    # cloneofsimo氏の実装ではmaskを平均で割って正規化していたが、
    # 一部のピクセルが異常に大きくなってlossがnanしたためやらない。
    stem = os.path.splitext(os.path.basename(file))[0]
    np.savez(os.path.join(output_dir, stem + ".npz"), mask)


def create_face_masks(directory, output_dir, cascade_path="lbpcascade_animeface.xml", num_workers=12, on_progress=None):
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"{cascade_path}が見つからないよ。https://github.com/nagadomi/lbpcascade_animeface から取ってきてね。")
    os.makedirs(output_dir, exist_ok=True)
    files = list_images(directory)

    worker = partial(_detect, directory=directory, output_dir=output_dir, cascade_path=cascade_path)
    with ProcessPoolExecutor(num_workers) as e:
        list(progress_iter(e.map(worker, files), total=len(files), desc="masks", on_progress=on_progress))

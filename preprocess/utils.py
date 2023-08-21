# ref:https://github.com/kohya-ss/sd-scripts/blob/main/finetune/tag_images_by_wd14_tagger.py

import numpy as np
import cv2

TAGGER_IMAGE_SIZE = 448
CLIP_VISION_IMAGE_SIZE = 224

def preprocess_for_tagger(image):
    image = image.convert("RGB")
    image = np.array(image)
    image = image[:, :, ::-1]  # RGB->BGR

    # pad to square
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=255)

    interp = cv2.INTER_AREA if size > TAGGER_IMAGE_SIZE else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (TAGGER_IMAGE_SIZE, TAGGER_IMAGE_SIZE), interpolation=interp)

    image = image.astype(np.float32)
    return image

def preprocess_for_clip_vision(image):
    image = image.convert("RGB")
    image = np.array(image)
    # pad to square
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=0)

    interp = cv2.INTER_AREA if size > CLIP_VISION_IMAGE_SIZE else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (CLIP_VISION_IMAGE_SIZE, CLIP_VISION_IMAGE_SIZE), interpolation=interp)

    image = image.astype(np.float32)
    return image


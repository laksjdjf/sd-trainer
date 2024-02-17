from modules.dataset import BaseDataset
import cv2
import os
import torch
import numpy as np
from torchvision import transforms

class CannyDataset(BaseDataset):
    def get_control(self, samples, dir="control"):
        images = []
        transform = transforms.ToTensor()
        for sample in samples:
            # ref https://qiita.com/kotai2003/items/662c33c15915f2a8517e
            image = cv2.imread(os.path.join(self.path, dir, sample + f".png"))
            med_val = np.median(image)
            sigma = 0.33  # 0.33
            min_val = int(max(0, (1.0 - sigma) * med_val))
            max_val = int(max(255, (1.0 + sigma) * med_val))
            image = cv2.Canny(image, threshold1 = min_val, threshold2 = max_val)
            image = image[:, :, None] # add channel
            image = np.concatenate([image]*3, axis=2) # grayscale to rgb
            images.append(transform(image))
        images_tensor = torch.stack(images).to(memory_format=torch.contiguous_format).float()
        return images_tensor
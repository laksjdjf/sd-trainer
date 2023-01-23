from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms
import glob

class SimpleDataset(Dataset):
    def __init__(self, path, size):
        self.file_list = []
        
        #画像リスト
        [self.file_list.extend(glob.glob(f'{path}' + '/*.' + e)) for e in ['jpg', 'jpeg', 'png', 'bmp', 'webp']]
        self.transform = transforms.Compose(
        [
        transforms.Resize(size),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ]
    )
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, i):
        img = Image.open(self.file_list[i])
        #txtを読み込む
        with open(os.path.splitext(self.file_list[i])[0] + ".txt" ,"r") as f:
            caption = f.read()
        return {"image":self.transform(img),"caption":caption}

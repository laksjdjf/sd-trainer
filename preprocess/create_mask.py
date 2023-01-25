import numpy as np
import os
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

path = "org"
output = "data"

def detect(file):
    classifier = cv2.CascadeClassifier('lbpcascade_animeface.xml')
    image = cv2.imread(os.path.join(path,file))
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray_image)
    
    #latentのサイズは画像サイズ÷8
    latent = np.zeros((image.shape[0]//8,image.shape[1]//8))
    #顔部分以外も完全に0にはしない
    latent += 0.05
    
    #顔部分は1
    for i, (x,y,w,h) in enumerate(faces):
        latent[y//8:(y+h)//8, x//8:(x+w)//8] = 1
    np.savez(os.path.join(output,file[:-4] + ".npz"),latent)
    
    #cloneofsimo氏の実装では、maskを平均で割ることで正規化していたが、
    #一部のピクセルが異常に大きくなってlossがnanしたためやらない。
            
def main():            
    files = [file for file in os.listdir(path) if "png" in file]
    with ProcessPoolExecutor(12) as e:
        results = list(tqdm(e.map(detect, files),total=len(files)))
    return 

if __name__ == "__main__":
    main() 

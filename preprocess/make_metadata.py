import json
import os
import sys
import glob
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

path = sys.argv[1]
processes = 8 if len(sys.argv) < 3 else int(sys.argv[2]) 
files = []
files = [file for file in os.listdir(path) if "png" in file]

file_splits = []
length = len(files)
for i in range(processes):
    start = (length//processes)*(i)
    end = length if i==(processes-1) else (length//processes)*(i+1)
    file_split = files[start:end]
    file_splits.append(file_split)

def make_dic(files):
    meta = {}
    for file in tqdm(files):
        try:
            img = Image.open(path+file)
        except:
            print(file)
            continue
        bucket = (img.width,img.height)

        if bucket in meta.keys():
            meta[bucket].append(file[:-4])
        else:
            meta[bucket] = [file[:-4]]
    return meta

#========計算処理========
with ProcessPoolExecutor(processes) as e:
    ret = e.map(make_dic, file_splits)
sms_multi = [r for r in ret]
#=======================

meta = {}
for i in range(0,processes):
    for key in sms_multi[i].keys():
        if str(key) in meta.keys():
            meta[str(key)] += sms_multi[i][key]
        else:
            meta[str(key)] = sms_multi[i][key]

with open(path+"buckets.json" ,"w") as f:
    json.dump(meta,f)

for key in meta:
    print(key,len(meta[key]))
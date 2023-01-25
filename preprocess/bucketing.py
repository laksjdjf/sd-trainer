import os
from tqdm import tqdm
import glob
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import json
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', '-i', type=str, required=True, help="元のデータセット")
parser.add_argument('--output_dir', '-o', type=str, required=False, help="保存先のディレクトリ")
parser.add_argument('--threads', '-p', required=False, default=12, type=int,help="プロセス数")
parser.add_argument('--resolution', '-r', required=False, default=768, type=int,help="bucketの解像度：64の倍数を推奨する")
parser.add_argument('--min_length', required=False, default=512, type=int,help="bucketの最小長")
parser.add_argument('--max_length', required=False, default=1024, type=int,help="bucketの最大長")
parser.add_argument('--max_ratio', required=False, default=2.0, type=float,help="最大アスペクト比（逆数が最小アスペクト比）")
args = parser.parse_args()

    
            
def make_buckets():
    #モデルの構造からして64の倍数が推奨される。（VAEで8分の1⇒UNetで8分の1）
    increment = 64
    #最大ピクセル数
    max_pixels = args.resolution*args.resolution
    
    #正方形は手動で追加
    buckets = set()
    buckets.add((args.resolution, args.resolution))
    
    #最小値から～
    width = args.min_length
    #～最大値まで
    while width <= args.max_length:
        #最大ピクセル数と最大長を越えない最大の高さ
        height = min(args.max_length, (max_pixels // width ) - (max_pixels // width ) % increment)
        ratio = width/height
        
        #アスペクト比が極端じゃなかったら追加、高さと幅入れ替えたものも追加。
        if 1/args.max_ratio <= ratio <= args.max_ratio:
            buckets.add((width, height))
            buckets.add((height, width))
        width += increment #幅を大きくして次のループへ
        
    #なんかアスペクト比順に並び替えたりしてるけどあんまり意味ない。
    buckets = list(buckets)
    ratios = [w/h for w,h in buckets]
    buckets = np.array(buckets)[np.argsort(ratios)]
    ratios = np.sort(ratios)
    return buckets, ratios

def resize_image(file):
    image = Image.open(file)
    image = image.convert("RGB")
    ratio = image.width / image.height
    ar_errors = ratios - ratio
    indice = np.argmin(np.abs(ar_errors)) #一番近いアスペクト比のインデックス
    bucket_width, bucket_height = buckets[indice]
    ar_error = ar_errors[indice]
    if ar_error <= 0: #幅＜高さなら高さを合わせる
        temp_width = int(image.width*bucket_height/image.height) 
        image = image.resize((temp_width,bucket_height)) #アスペクト比を変えずに高さだけbucketに合わせる
        left = (temp_width - bucket_width) / 2 #切り取り境界左側
        right = bucket_width + left #切り取り境界右側
        image = image.crop((left,0,right,bucket_height)) #左右切り取り
    else: #幅高さを逆にしたもの
        temp_height = int(image.height*bucket_width/image.width)
        image = image.resize((bucket_width,temp_height))
        upper = (temp_height - bucket_height) / 2
        lower = bucket_height + upper
        image = image.crop((0,upper,bucket_width,lower))
    image.save(os.path.join(args.output_dir,os.path.basename(file)))
    return [os.path.splitext(os.path.basename(file))[0],str((bucket_width,bucket_height))]

def main():
    
    files = []
    [files.extend(glob.glob(f'{args.input_dir}' + '/*.' + e)) for e in ['jpg', 'jpeg', 'png', 'bmp', 'webp']]
    with ProcessPoolExecutor(8) as e:
        results = list(tqdm(e.map(resize_image, files),total=len(files)))
    
    #メタデータを書き込む（どのファイルがどのbucketにあるかを保存しておく）
    meta = {}
    print(results)
    for file,bucket in results:
        if bucket in meta:
            meta[bucket].append(file)
        else:
            meta[bucket] = [file]
    for key in meta:
        print(f"{key}: {len(meta[key])}個")
    with open(os.path.join(args.output_dir,"buckets.json"), "w") as f:
        json.dump(meta,f)
    return

        
if __name__ == '__main__':
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    buckets,ratios = make_buckets()
    main()

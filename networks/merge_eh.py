import argparse
import os

from diffusers import StableDiffusionPipeline

from eh import EHNetwork
###コマンドライン引数#########################################################################
parser = argparse.ArgumentParser(description='まーじいーえいち')
parser.add_argument('--model', type=str, required=True, help='元モデルパス（diffusers）')
parser.add_argument('--output', type=str, required=True, help='出力先')
parser.add_argument('--eh', type=str,required = True, help='ehのpt')
parser.add_argument('--num_groups', type=int,default = 4, help='グループ数')
parser.add_argument('--multiplier', type=float,default = 1.0, help='重み付け')
############################################################################################

def main(args):
    pipe = StableDiffusionPipeline.from_pretrained(
            args.model,
            feature_extractor = None,
            safety_checker = None
    )
    
    EHNetwork(pipe.unet,args.num_groups, multiplier = args.multiplier, merge=True, resume=args.eh)
    pipe.save_pretrained(f'{args.output}')
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

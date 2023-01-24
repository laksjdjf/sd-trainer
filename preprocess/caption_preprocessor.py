#This code is based on 
import re
import ast
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os

ASTOLFO_DELETE_TAGS = set(["long_hair","pink_hair","single_braid","male_focus","purple_eyes","1boy","multicolored_hair",
                  "hair_intakes","streaked_hair","white_hair","hair_between_eyes","simple_background","white_background",":d",";d"])

###コマンドライン引数#########################################################################
parser = argparse.ArgumentParser(description='StableDiffusionの訓練コード')
parser.add_argument('--dataset', type=str, required=True, help='datasetパス')
parser.add_argument('--use_origin_tag',action = "store_true", help='danbooruのタグを使う')
parser.add_argument('--astolfo',action = "store_true", help='astolfo学習用')
args = parser.parse_args()
############################################################################################

class CaptionProcessor(object):
    def __init__(self):
        pass
    def clean(self, text: str):
        if args.astolfo:
            text = " ".join(list(set(text.split(" ")) - ASTOLFO_DELETE_TAGS))
        #カッコ内を全て削除する
        text = ' '.join(set([i.lstrip('_').rstrip('_') for i in re.sub(r'\([^)]*\)', '', text).split(' ')])).lstrip().rstrip()
        
        #_をスペースにしてコンマでタグを区切る
        text = ', '.join([i.replace('_', ' ') for i in text.split(' ')]).lstrip(', ').rstrip(', ')
        return text
    
    def get_character(self, val_dict, key, prepend_space = False, append_comma = False):
        if key not in val_dict:
            return ''
        #先頭のスペース
        space = ' ' if prepend_space else ''
        #最後のコンマ
        comma = ',' if append_comma else ''
        
        #キャラクターの衣装タグ
        costume = str(set(re.findall("(?<=\().+?(?=\))", val_dict[key])))
        
        #最終再臨でない場合分ける
        if args.astolfo:
            if "saber" in costume and "third_ascension" not in costume:
                costume = costume.replace("'saber'","'saber', 'first_saint_graph'")
        text = self.clean(val_dict[key]) + ", " + re.sub("[{}']","",costume).replace("_"," ")
        return space + text.rstrip(", ") + comma

    def get_key(self, val_dict, key, prepend_space = False, append_comma = False):
        if key not in val_dict:
            return ''
        #先頭のスペース
        space = ' ' if prepend_space else ''
        #最後のコンマ
        comma = ',' if append_comma else ''
        return space + self.clean(val_dict[key]) + comma
    
    #wd14に合わせたメタタグ
    def get_metatag(self,caption_data):
        if "rating" in caption_data.keys():
            rate = caption_data["rating"]
            if rate == "g":
                rating = "safe, "
            elif rate == "s":
                rating = "questionable, "
            else:
                rating = "nsfw, "
        else:
            rating = ""
                
        if "score" in caption_data.keys():
            point = caption_data["score"]
            if point > 150:
                score = "masterpiece, "
            elif point > 100:
                score = "best quality, "
            elif point > 75:
                score = "high quality, "
            elif point > 25:
                score = "medium quality, "
            elif point < -5:
                score = "worst quality, "
            elif point < 0:
                score = "low quality, "
            else:
                score = ""
        else:
            score = ""
            
        metadata = ""
        if "tag_string_meta" in caption_data.keys():
            meta = caption_data["tag_string_meta"]
            if "highres" in meta:
                metadata = "highres, "
                
            if "absurdres" in meta:
                metadata += "absurdres, "
            
            if "lowres" in meta:
                metadata += "lowres. "

        if "is_deleted" in caption_data.keys():
            if caption_data["is_deleted"] == "True":
                metadata += "deleted, "
        return score + rating + metadata

    def __call__(self, caption_data):
        character = self.get_character(caption_data, 'tag_string_character', False, True)
        copyright = self.get_key(caption_data, 'tag_string_copyright', True, True)
        artist = self.get_key(caption_data, 'tag_string_artist', True, True)
        if args.use_origin_tag:
            general = self.get_key(caption_data, 'tag_string_general', True, False)
        else:
            general = self.get_key(caption_data, 'tagger', True, False)
        data = self.get_metatag(caption_data)
        
        return f'{data}{character}{copyright}{artist}{general}'.lstrip().rstrip(',')

def preprocess(file):
    with open(os.path.join(args.dataset,file), 'r', encoding='UTF-8') as f:
        caption = processor(ast.literal_eval(f.read().replace('"lazy"','lazy'))) #wd14tagger特有のバグ回避
    with open(os.path.join(args.dataset,file[:-4] + ".caption"),"w") as f:
        f.write(caption)
    return 

def main():            
    files = [file for file in os.listdir(args.dataset) if "txt" in file]
    with ProcessPoolExecutor(8) as e:
        results = list(tqdm(e.map(preprocess, files),total=len(files)))
    return 

if __name__ == "__main__":
    processor = CaptionProcessor()
    main()
        

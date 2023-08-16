import json
import sys
dict_path = sys.argv[1]
output_path = sys.argv[2]

with open(dict_path,"r") as f:
    meta = json.load(f)

dic = {}

count = 0
for key, value in meta.items():
    if "image_width" in value:
        dic[key] = {
            "original_width": int(value["image_width"]),
            "original_height": int(value["image_height"]),
        }
        count += 1

print(count)
with open(output_path,"w") as f:
    json.dump(dic,f)
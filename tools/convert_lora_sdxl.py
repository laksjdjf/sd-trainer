# python convert_lora_sdxl.py <input_file_name> <output_file_name> <sd2diff or diff2sd>

import torch
from safetensors.torch import load_file
from safetensors.torch import save_file
import os
import argparse

def load(file):
    if os.path.splitext(file)[1] == ".safetensors":
        return load_file(file)
    else:
        return torch.load(file)
    
def save(state_dict, file):
    if os.path.splitext(file)[1] == ".safetensors":
        return save_file(state_dict, file)
    else:
        return torch.save(state_dict, file)

# make mapping list
unet_conversion_map_layer = []
for i in range(3): # num_blocks is 3 in sdxl
    # loop over downblocks/upblocks
    for j in range(2):
        # loop over resnets/attentions for downblocks
        hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
        sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
        unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

        if i < 3:
            # no attention layers in down_blocks.3
            hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
            sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
            unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

    for j in range(3):
        # loop over resnets/attentions for upblocks
        hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
        sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
        unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

        # if i > 0: commentout for sdxl 
        # no attention layers in up_blocks.0
        hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
        sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
        unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

    if i < 3:
        # no downsample in down_blocks.3
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv"
        sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op"
        unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

        # no upsample in up_blocks.3
        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"output_blocks.{3*i + 2}.{2}." #change for sdxl
        unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

for j in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{j}."
    sd_mid_res_prefix = f"middle_block.{2*j}."
    unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))
    
unet_conversion_map_resnet = [
    # (stable-diffusion, HF Diffusers)
    ("in_layers.0", "norm1"),
    ("in_layers.2", "conv1"),
    ("out_layers.0", "norm2"),
    ("out_layers.3", "conv2"),
    ("emb_layers.1", "time_emb_proj"),
    ("skip_connection", "conv_shortcut"),
]

# mapping dict: key=stabilityAI_key, value=diffusers_key
sd2diff_map = {}
for k,v in unet_conversion_map_layer:
    sd_key = k.replace(".","_")
    diff_key = v.replace(".","_")
    sd2diff_map[sd_key] = diff_key

for k,v in unet_conversion_map_resnet:
    sd_key = k.replace(".","_")
    diff_key = v.replace(".","_")
    sd2diff_map[sd_key] = diff_key

'''
sd2diff_map = {
    'input_blocks_1_0_': 'down_blocks_0_resnets_0_',
    'input_blocks_1_1_': 'down_blocks_0_attentions_0_',
    'input_blocks_2_0_': 'down_blocks_0_resnets_1_',
    'input_blocks_2_1_': 'down_blocks_0_attentions_1_',
    'output_blocks_0_0_': 'up_blocks_0_resnets_0_',
    'output_blocks_0_1_': 'up_blocks_0_attentions_0_',
    'output_blocks_1_0_': 'up_blocks_0_resnets_1_',
    'output_blocks_1_1_': 'up_blocks_0_attentions_1_',
    'output_blocks_2_0_': 'up_blocks_0_resnets_2_',
    'output_blocks_2_1_': 'up_blocks_0_attentions_2_',
    'input_blocks_3_0_op': 'down_blocks_0_downsamplers_0_conv',
    'output_blocks_2_2_': 'up_blocks_0_upsamplers_0_',
    'input_blocks_4_0_': 'down_blocks_1_resnets_0_',
    'input_blocks_4_1_': 'down_blocks_1_attentions_0_',
    'input_blocks_5_0_': 'down_blocks_1_resnets_1_',
    'input_blocks_5_1_': 'down_blocks_1_attentions_1_',
    'output_blocks_3_0_': 'up_blocks_1_resnets_0_',
    'output_blocks_3_1_': 'up_blocks_1_attentions_0_',
    'output_blocks_4_0_': 'up_blocks_1_resnets_1_',
    'output_blocks_4_1_': 'up_blocks_1_attentions_1_',
    'output_blocks_5_0_': 'up_blocks_1_resnets_2_',
    'output_blocks_5_1_': 'up_blocks_1_attentions_2_',
    'input_blocks_6_0_op': 'down_blocks_1_downsamplers_0_conv',
    'output_blocks_5_2_': 'up_blocks_1_upsamplers_0_',
    'input_blocks_7_0_': 'down_blocks_2_resnets_0_',
    'input_blocks_7_1_': 'down_blocks_2_attentions_0_',
    'input_blocks_8_0_': 'down_blocks_2_resnets_1_',
    'input_blocks_8_1_': 'down_blocks_2_attentions_1_',
    'output_blocks_6_0_': 'up_blocks_2_resnets_0_',
    'output_blocks_6_1_': 'up_blocks_2_attentions_0_',
    'output_blocks_7_0_': 'up_blocks_2_resnets_1_',
    'output_blocks_7_1_': 'up_blocks_2_attentions_1_',
    'output_blocks_8_0_': 'up_blocks_2_resnets_2_',
    'output_blocks_8_1_': 'up_blocks_2_attentions_2_',
    'input_blocks_9_0_op': 'down_blocks_2_downsamplers_0_conv',
    'output_blocks_8_2_': 'up_blocks_2_upsamplers_0_',
    'middle_block_1_': 'mid_block_attentions_0_',
    'middle_block_0_': 'mid_block_resnets_0_',
    'middle_block_2_': 'mid_block_resnets_1_',
    'in_layers_0': 'norm1',
    'in_layers_2': 'conv1',
    'out_layers_0': 'norm2',
    'out_layers_3': 'conv2',
    'emb_layers_1': 'time_emb_proj',
    'skip_connection': 'conv_shortcut',
}
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sdxl版LoRAの変換コード')
    parser.add_argument('input', type=str, help='変換元LoRAのパス')
    parser.add_argument('output', type=str, help='変換先LoRAのパス')
    parser.add_argument('type', type=str, choices=['sd2diff', 'diff2sd'], help='stabilityAIからdiffusersはsd2diff、diffusersからstabilityAIはdiff2sd')
    parser.add_argument('--verbose', '-v', action='store_true', help='詳細出力を有効にする')
    args = parser.parse_args()

    state_dict = load(args.input)
    new_state_dict = {}
    for key in state_dict:
        new_key = key
        for k,v in sd2diff_map.items():
            if args.type == 'sd2diff':
                new_key = new_key.replace(k,v)
            elif args.type == 'diff2sd':
                new_key = new_key.replace(v,k)
        if args.verbose and "alpha" in key: # 長くなるのでalphaのみ表示
            print(f"replace {key.replace('.alpha','')} to {new_key.replace('.alpha','')}")
        new_state_dict[new_key] = state_dict[key]
    save(new_state_dict, args.output)

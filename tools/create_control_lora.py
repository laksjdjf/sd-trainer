import torch
from tqdm import tqdm
CLAMP_QUANTILE = 0.99

def make_unet_conversion_map():
    unet_conversion_map_layer = []
    # unet
    # https://github.com/kohya-ss/sd-scripts/blob/2d7389185c021bc527b414563c245c5489d6328a/library/sdxl_model_util.py#L293
    for i in range(3):  # num_blocks is 3 in sdxl
        # loop over downblocks/upblocks
        for j in range(2):
            # loop over resnets/attentions for downblocks
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

            if i < 3:
                # no attention layers in down_blocks.3
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}"
                sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1"
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

        for j in range(3):
            # loop over resnets/attentions for upblocks
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
            unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

            # if i > 0: commentout for sdxl
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}"
            sd_up_atn_prefix = f"output_blocks.{3*i + j}.1"
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

        if i < 3:
            # no downsample in down_blocks.3
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv"
            sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op"
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

            # no upsample in up_blocks.3
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0"
            sd_upsample_prefix = f"output_blocks.{3*i + 2}.{2}"  # change for sdxl
            unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

    hf_mid_atn_prefix = "mid_block.attentions.0"
    sd_mid_atn_prefix = "middle_block.1"
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

    unet_conversion_map = []
    for sd, hf in unet_conversion_map_layer:
        if "resnets" in hf:
            for sd_res, hf_res in unet_conversion_map_resnet:
                unet_conversion_map.append((sd + sd_res, hf + hf_res))
        else:
            unet_conversion_map.append((sd, hf))

    for j in range(2):
        hf_time_embed_prefix = f"time_embedding.linear_{j+1}"
        sd_time_embed_prefix = f"time_embed.{j*2}"
        unet_conversion_map.append((sd_time_embed_prefix, hf_time_embed_prefix))

    for j in range(2):
        hf_label_embed_prefix = f"add_embedding.linear_{j+1}"
        sd_label_embed_prefix = f"label_emb.0.{j*2}"
        unet_conversion_map.append((sd_label_embed_prefix, hf_label_embed_prefix))

    unet_conversion_map.append(("input_blocks.0.0", "conv_in"))

    # controlnet
    # created by chatgpt
    mapping_dict = {
        "input_hint_block.0": "controlnet_cond_embedding.conv_in",
        # 以下、input_hint_blockの残りのマッピングを定義
    }

    # input_hint_blockのマッピングを追加
    orig_index = 2  # 既に0番目は上で定義されているため2から開始
    diffusers_index = 0
    while diffusers_index < 6:
        mapping_dict[f"input_hint_block.{orig_index}"] = f"controlnet_cond_embedding.blocks.{diffusers_index}"
        diffusers_index += 1
        orig_index += 2

    # 最後のconv_outのマッピングを追加
    mapping_dict[f"input_hint_block.{orig_index}"] = "controlnet_cond_embedding.conv_out"

    # down blocksとmid blockのマッピングを追加
    num_input_blocks = 12
    for i in range(num_input_blocks):
        mapping_dict[f"zero_convs.{i}.0"] = f"controlnet_down_blocks.{i}"

    mapping_dict["middle_block_out.0"] = "controlnet_mid_block"

    mapping_dict.update({t[0]:t[1] for t in unet_conversion_map})
    
    return mapping_dict

def convert_key(key, mapping_dict, mode="diff2sgm"):
    new_key = key
    for k,v in mapping_dict.items():
        if mode == "diff2sgm":
            new_key = new_key.replace(v, k)
        elif mode == "sgm2diff":
            new_key = new_key.replace(k, v)
        else:
            raise ValueError("mode should be 'diff2sgm' or 'sgm2diff'")
    return new_key

def create_lora(controlnet, unet, rank=128, device="cuda", dtype=torch.float16):
    mapping_dict = make_unet_conversion_map()
    new_dic = {}
    for diff_key, value in tqdm(controlnet.items()):
        sgm_key = convert_key(diff_key, mapping_dict)
        if "controlnet" in diff_key:
            new_dic[sgm_key] = value
        elif value.dim() == 1:
            new_dic[sgm_key] = value
        else:
            pretrained_value = unet[diff_key]
            up, down = svd_extract(pretrained_value, value, rank=rank, device=device, dtype=dtype)
            new_dic[sgm_key.replace("weight", "up")] = up
            new_dic[sgm_key.replace("weight", "down")] = down
    new_dic["lora_controlnet"] = torch.tensor([])
    return new_dic
            
@torch.no_grad()
def svd_extract(pretrained, finetuned, rank, device="cuda", dtype=torch.float16):
    
    weight = finetuned.to(device, dtype=torch.float32) - pretrained.to(device, dtype=torch.float32)

    rank = min(rank, weight.shape[0], weight.shape[1])

    if weight.dim() == 4:
        weight = weight.reshape(weight.shape[0], -1)
    U, S, Vh = torch.linalg.svd(weight)

    U = U[:, :rank]
    S = S[:rank]
    U = U @ torch.diag(S)

    Vh = Vh[:rank, :]

    dist = torch.cat([U.flatten(), Vh.flatten()])
    hi_val = torch.quantile(dist, CLAMP_QUANTILE)
    low_val = -hi_val

    U = U.clamp(low_val, hi_val)
    Vh = Vh.clamp(low_val, hi_val)

    if pretrained.dim() == 4:
        U = U.reshape(pretrained.shape[0], rank, 1, 1)
        Vh = Vh.reshape(rank, pretrained.shape[1], pretrained.shape[2], pretrained.shape[3])

    up = U.to("cpu", dtype=dtype)
    down = Vh.to("cpu", dtype=dtype)

    return up, down

if __name__ == "__main__":
    import argparse
    from safetensors.torch import load_file, save_file
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlnet", "-c", type=str, required=True)
    parser.add_argument("--unet", "-u", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--rank", "-r", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="torch.float16")
    args = parser.parse_args()

    controlnet = load_file(args.controlnet)
    print("controlnet loaded")
    unet = load_file(args.unet)
    print("unet loaded")

    control_lora = create_lora(controlnet, unet, rank=args.rank, device=args.device, dtype=eval(args.dtype))
    print("saving...")
    save_file(control_lora, args.output)

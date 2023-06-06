# Test script
import torch
import sys

input = sys.argv[1]
output = sys.argv[2]

ckpt = torch.load(f"{input}/diffusion_pytorch_model.bin", map_location=torch.device('cpu'))

new_checkpoint = ckpt
num_input_blocks = 12

orig_index = 0
unet_state_dict = {}

unet_state_dict[f"input_hint_block.{orig_index}.weight"] = new_checkpoint.pop(
    "controlnet_cond_embedding.conv_in.weight"
)
unet_state_dict[f"input_hint_block.{orig_index}.bias"] = new_checkpoint.pop(
    "controlnet_cond_embedding.conv_in.bias"
)

orig_index += 2

diffusers_index = 0

while diffusers_index < 6:
    unet_state_dict[f"input_hint_block.{orig_index}.weight"] = new_checkpoint.pop(
        f"controlnet_cond_embedding.blocks.{diffusers_index}.weight"
    )
    unet_state_dict[f"input_hint_block.{orig_index}.bias"] = new_checkpoint.pop(
        f"controlnet_cond_embedding.blocks.{diffusers_index}.bias"
    )
    diffusers_index += 1
    orig_index += 2

unet_state_dict[f"input_hint_block.{orig_index}.weight"] = new_checkpoint.pop(
    "controlnet_cond_embedding.conv_out.weight"
)
unet_state_dict[f"input_hint_block.{orig_index}.bias"] = new_checkpoint.pop(
    "controlnet_cond_embedding.conv_out.bias"
)

# down blocks
for i in range(num_input_blocks):
    unet_state_dict[f"zero_convs.{i}.0.weight"] = new_checkpoint.pop(f"controlnet_down_blocks.{i}.weight")
    unet_state_dict[f"zero_convs.{i}.0.bias"] = new_checkpoint.pop(f"controlnet_down_blocks.{i}.bias")

# mid block
unet_state_dict["middle_block_out.0.weight"] = new_checkpoint.pop("controlnet_mid_block.weight")
unet_state_dict["middle_block_out.0.bias"] = new_checkpoint.pop("controlnet_mid_block.bias")

# Add "control_model." to each key
unet_state_dict = {f"control_model.{k}": v for k, v in unet_state_dict.items()}

unet_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
]

unet_conversion_map_resnet = [
    # (stable-diffusion, HF Diffusers)
    ("in_layers.0", "norm1"),
    ("in_layers.2", "conv1"),
    ("out_layers.0", "norm2"),
    ("out_layers.3", "conv2"),
    ("emb_layers.1", "time_emb_proj"),
    ("skip_connection", "conv_shortcut"),
]

unet_conversion_map_layer = []
# hardcoded number of downblocks and resnets/attentions...
# would need smarter logic for other networks.
for i in range(4):
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

        if i > 0:
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

    if i < 3:
        # no downsample in down_blocks.3
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
        sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
        unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

        # no upsample in up_blocks.3
        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"output_blocks.{3*i + 2}.{1 if i == 0 else 2}."
        unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

for j in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{j}."
    sd_mid_res_prefix = f"middle_block.{2*j}."
    unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))


def convert_unet_state_dict(unet_state_dict):
    # buyer beware: this is a *brittle* function,
    # and correct output requires that all of these pieces interact in
    # the exact order in which I have arranged them.
    mapping = {k: k for k in unet_state_dict.keys()}
    for sd_name, hf_name in unet_conversion_map:
        if hf_name in mapping:
            mapping[hf_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in unet_conversion_map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, hf_part in unet_conversion_map_layer:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    new_state_dict = {"control_model."+v:unet_state_dict[k] for k, v in mapping.items()}
    return new_state_dict
    
control = convert_unet_state_dict(ckpt)
for key in control:
    unet_state_dict[key] = control[key]
    
for key in unet_state_dict:
    #print(unet_state_dict[key].mean())
    unet_state_dict[key] = unet_state_dict[key].half()

torch.save(unet_state_dict,output)

import argparse
import os

from safetensors.torch import load_file, save_file


SUPPORTED_EXTERNAL_MODULES = (
    "self_attn_q_proj",
    "self_attn_k_proj",
    "self_attn_v_proj",
    "self_attn_output_proj",
    "cross_attn_q_proj",
    "cross_attn_k_proj",
    "cross_attn_v_proj",
    "cross_attn_output_proj",
    "mlp_layer1",
    "mlp_layer2",
)

SD_TO_EXTERNAL_MODULES = {
    "attn1_to_q": "self_attn_q_proj",
    "attn1_to_k": "self_attn_k_proj",
    "attn1_to_v": "self_attn_v_proj",
    "attn1_to_out_0": "self_attn_output_proj",
    "attn2_to_q": "cross_attn_q_proj",
    "attn2_to_k": "cross_attn_k_proj",
    "attn2_to_v": "cross_attn_v_proj",
    "attn2_to_out_0": "cross_attn_output_proj",
    "ff_net_0_proj": "mlp_layer1",
    "ff_net_2": "mlp_layer2",
    "norm1_linear_1": "adaln_modulation_self_attn_1",
    "norm1_linear_2": "adaln_modulation_self_attn_2",
    "norm2_linear_1": "adaln_modulation_cross_attn_1",
    "norm2_linear_2": "adaln_modulation_cross_attn_2",
    "norm3_linear_1": "adaln_modulation_mlp_1",
    "norm3_linear_2": "adaln_modulation_mlp_2",
}

EXTERNAL_TO_SD_MODULES = {value: key for key, value in SD_TO_EXTERNAL_MODULES.items()}


def is_supported_external_stem(stem):
    if not stem.startswith("lora_unet_blocks_"):
        return True
    return any(stem.endswith("_" + module) for module in SUPPORTED_EXTERNAL_MODULES)


def replace_module_suffix(stem, mapping):
    for src, dst in mapping.items():
        suffix = "_" + src
        if stem.endswith(suffix):
            return stem[: -len(suffix)] + "_" + dst
    return stem


def external_to_sd_trainer(key):
    if key.startswith("lora_unet_blocks_"):
        key = "lora_transformer_core_transformer_" + key[len("lora_unet_") :]
        return replace_module_suffix(key, EXTERNAL_TO_SD_MODULES)
    return key


def sd_trainer_to_external(key):
    if key.startswith("lora_transformer_core_transformer_blocks_"):
        key = "lora_unet_" + key[len("lora_transformer_core_transformer_") :]
        return replace_module_suffix(key, SD_TO_EXTERNAL_MODULES)
    if key.startswith("lora_transformer_core_blocks_"):
        key = "lora_unet_" + key[len("lora_transformer_core_") :]
        return replace_module_suffix(key, SD_TO_EXTERNAL_MODULES)
    return key


def convert_key(key, mode):
    stem, suffix = key.split(".", 1)
    if mode == "external2sd":
        stem = external_to_sd_trainer(stem)
    elif mode == "sd2external":
        stem = sd_trainer_to_external(stem)
    else:
        raise ValueError("mode must be external2sd or sd2external")
    return stem + "." + suffix


def main():
    parser = argparse.ArgumentParser(description="Convert Anima LoRA keys between external lora_unet_blocks_* and sd-trainer lora_transformer_core_blocks_* formats.")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("mode", choices=["external2sd", "sd2external"])
    parser.add_argument("--drop-unsupported", action="store_true", help="drop converted keys outside the common Anima LoRA attn/mlp set")
    args = parser.parse_args()

    state_dict = load_file(args.input)
    converted = {}
    for key, value in state_dict.items():
        new_key = convert_key(key, args.mode)
        stem = new_key.split(".", 1)[0]
        if args.mode == "sd2external" and args.drop_unsupported and not is_supported_external_stem(stem):
            continue
        converted[new_key] = value

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_file(converted, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

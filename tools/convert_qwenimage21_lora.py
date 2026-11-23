"""Convert sd-trainer Qwen-Image-2.1 LoRA keys to ComfyUI's transformer format."""
import argparse
import re
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import load_file, save_file

MODULES = {
    "attn_to_q": "attn.to_q", "attn_to_k": "attn.to_k",
    "attn_to_v": "attn.to_v", "attn_to_out_0": "attn.to_out.0",
    "img_mlp_gate_layer": "img_mlp.gate_layer",
    "img_mlp_proj": "img_mlp.proj", "img_mlp_out": "img_mlp.out",
}
PATTERN = re.compile(r"lora_transformer_transformer_blocks_(\d+)_(.+)\.(alpha|lora_down\.weight|lora_up\.weight)$")


def convert(state):
    output = {}
    for key, tensor in state.items():
        match = PATTERN.fullmatch(key)
        if match is None or match[2] not in MODULES:
            raise ValueError(f"Unsupported Qwen-Image-2.1 LoRA key: {key}")
        block, module, suffix = match.groups()
        target = f"transformer.transformer_blocks.{block}.{MODULES[module]}.{suffix}"
        if target in output:
            raise ValueError(f"Duplicate output key: {target}")
        output[target] = tensor
    if not output:
        raise ValueError("The input LoRA is empty.")
    for key in output:
        stem = key.split(".lora_")[0] if ".lora_" in key else key.removesuffix(".alpha")
        required = [stem + suffix for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha")]
        if not all(k in output for k in required):
            raise ValueError(f"Incomplete LoRA module: {stem}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Use a separate output file to preserve the training checkpoint.")
    if args.output.exists():
        parser.error("Output already exists; choose a new output filename.")
    with safe_open(args.input, framework="pt") as f:
        metadata = f.metadata() or {}
    result = convert(load_file(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(result, args.output, metadata=metadata)
    print(f"Saved {len(result)} tensors to {args.output}")


if __name__ == "__main__":
    main()

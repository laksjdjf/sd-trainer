"""Convert sd-trainer MOSS LoRA keys to ComfyUI's lora_unet format."""
import argparse
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


PATTERN = re.compile(
    r"lora_transformer_(layers_\d+_proj_(?:q|k|v|o|ffn_up_gate|ffn_down)|"
    r"text_embedder_blocks_\d+_(?:qkv|proj|ffn_gate|ffn_up|ffn_down))"
    r"\.(alpha|lora_down\.weight|lora_up\.weight)$"
)


def convert(state):
    output = {}
    for key, tensor in state.items():
        match = PATTERN.fullmatch(key)
        if match is None:
            raise ValueError(f"Unsupported MOSS-Image LoRA key: {key}")
        target = f"lora_unet_{match[1]}.{match[2]}"
        output[target] = tensor
    if not output:
        raise ValueError("The input LoRA is empty.")
    stems = {key.split(".", 1)[0] for key in output}
    for stem in stems:
        required = [stem + suffix for suffix in (".alpha", ".lora_down.weight", ".lora_up.weight")]
        if not all(key in output for key in required):
            raise ValueError(f"Incomplete LoRA module: {stem}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing converted output file.")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files.")
    if args.output.exists() and not args.overwrite:
        parser.error("Output already exists. Use --overwrite to update it.")
    with safe_open(args.input, framework="pt") as f:
        metadata = f.metadata() or {}
    result = convert(load_file(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(result, args.output, metadata=metadata)
    print(f"Saved {len(result) // 3} LoRA modules to {args.output}")


if __name__ == "__main__":
    main()

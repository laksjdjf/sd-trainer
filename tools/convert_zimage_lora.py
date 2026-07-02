#!/usr/bin/env python3
"""
Convert LoRA weights exported for the z_x4k base model into the naming/layout
used by z_image_turbo_childrens_drawings.safetensors.

The conversion rewrites tensor keys from the
`lora_transformer_*_{block}.(lora_down|lora_up|alpha)` convention into
`diffusion_model.<section>.layers.<idx>.<block>.lora_(A|B).weight`. Alpha scalars
are folded into `lora_B` (scale = alpha / rank) because the reference format
does not store alpha separately.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from safetensors import numpy as stnp

SECTION_PREFIX = {
    "layers": "diffusion_model.layers.{layer}",
    "context_refiner": "diffusion_model.context_refiner.{layer}",
    "noise_refiner": "diffusion_model.noise_refiner.{layer}",
}

BLOCK_NAME_MAP = {
    "adaLN_modulation_0": "adaLN_modulation.0",
    "attention_to_k": "attention.to_k",
    "attention_to_out_0": "attention.to_out.0",
    "attention_to_q": "attention.to_q",
    "attention_to_v": "attention.to_v",
    "feed_forward_w1": "feed_forward.w1",
    "feed_forward_w2": "feed_forward.w2",
    "feed_forward_w3": "feed_forward.w3",
}


def parse_key(name: str) -> Tuple[str, int, str, str]:
    """
    Returns (section, layer_idx, block_name, tensor_kind).

    Example:
        lora_transformer_layers_0_attention_to_k.lora_down.weight
        -> ('layers', 0, 'attention_to_k', 'lora_down.weight')
    """
    if not name.startswith("lora_transformer_"):
        raise ValueError(f"Unsupported key prefix: {name}")

    remainder = name[len("lora_transformer_") :]
    for section in sorted(SECTION_PREFIX, key=len, reverse=True):
        prefix = section + "_"
        if remainder.startswith(prefix):
            after_section = remainder[len(prefix) :]
            break
    else:
        raise ValueError(f"Unknown section in key: {name}")

    try:
        layer_str, after_layer = after_section.split("_", 1)
    except ValueError as exc:
        raise ValueError(f"Malformed key: {name}") from exc

    block, tensor_kind = after_layer.split(".", 1)
    return section, int(layer_str), block, tensor_kind


def build_modules(
    tensors: Dict[str, np.ndarray],
) -> Dict[Tuple[str, int, str], Dict[str, np.ndarray]]:
    modules: Dict[Tuple[str, int, str], Dict[str, np.ndarray]] = defaultdict(dict)
    for name, tensor in tensors.items():
        section, layer_idx, block, kind = parse_key(name)
        modules[(section, layer_idx, block)][kind] = tensor
    return modules


def convert_modules(
    modules: Dict[Tuple[str, int, str], Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    converted: Dict[str, np.ndarray] = {}

    for (section, layer_idx, block), parts in sorted(modules.items()):
        if section not in SECTION_PREFIX:
            raise ValueError(f"Unknown section '{section}' in key ({section}, {layer_idx}, {block})")
        if block not in BLOCK_NAME_MAP:
            raise ValueError(f"Unknown block '{block}' in key ({section}, {layer_idx}, {block})")

        base_prefix = SECTION_PREFIX[section].format(layer=layer_idx)
        block_name = BLOCK_NAME_MAP[block]

        lora_down = parts.get("lora_down.weight")
        lora_up = parts.get("lora_up.weight")
        if lora_down is None or lora_up is None:
            raise ValueError(f"Missing lora_down/up for module ({section}, {layer_idx}, {block})")

        rank = lora_down.shape[0]
        if rank == 0:
            raise ValueError(f"Rank is zero for module ({section}, {layer_idx}, {block})")
        alpha_value = parts.get("alpha")
        alpha = float(alpha_value.reshape(())) if alpha_value is not None else float(rank)
        scale = alpha / float(rank)

        converted[f"{base_prefix}.{block_name}.lora_A.weight"] = lora_down
        converted[f"{base_prefix}.{block_name}.lora_B.weight"] = lora_up * scale

    return converted


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Convert z_x4k LoRA weights into the z_image_turbo layout."
    )
    parser.add_argument("input_path", type=Path, help="z_x4k.safetensors file")
    parser.add_argument(
        "output_path", type=Path, help="Destination path for the converted safetensors file"
    )
    args = parser.parse_args(argv)

    tensors = stnp.load_file(str(args.input_path))
    modules = build_modules(tensors)
    converted = convert_modules(modules)
    stnp.save_file(converted, str(args.output_path))

    print(
        f"Converted {len(modules)} modules "
        f"({len(converted)} tensors) -> {args.output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

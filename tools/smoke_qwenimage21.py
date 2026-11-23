"""Small random-weight GPU smoke test; downloads only the official processor.
Run: CUDA_VISIBLE_DEVICES=1 python -m tools.smoke_qwenimage21 [--convrot forward|backward]
This checks integration, not pretrained image quality or full-model memory use.
"""
import argparse
from pathlib import Path
import torch
import numpy as np
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from diffusers import AutoencoderKLQwenImage21, QwenImage21Transformer2DModel, FlowMatchEulerDiscreteScheduler
from modules.text.qwenimage21 import QwenImage21TextModel
from modules.diffusion.qwenimage21 import QwenImage21DiffusionModel
from modules.qwenimage21_scheduler import QwenImage21Scheduler
from modules.trainer import BaseTrainer
from modules.config import TrainerConfig
from omegaconf import OmegaConf
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--convrot", choices=["forward", "backward"], default=None)
parser.add_argument("--te-device", default="cpu")
args = parser.parse_args()

torch.manual_seed(7)
torch.set_num_threads(2)
Path("outputs/qwen21_smoke").mkdir(parents=True, exist_ok=True)
processor = Qwen3VLProcessor.from_pretrained("Qwen/Qwen-Image-2.1", subfolder="processor")
config = Qwen3VLConfig(
    text_config=dict(vocab_size=151936, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=16,
                     rope_parameters={"rope_type": "default", "rope_theta": 500000.,
                                      "mrope_section": [2, 3, 3]}),
    vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                       out_hidden_size=32, deepstack_visual_indexes=[]),
)
text = QwenImage21TextModel(processor, Qwen3VLForConditionalGeneration(config))
vae = AutoencoderKLQwenImage21(base_dim=4, decoder_base_dim=4, z_dim=64,
                              dim_mult=[1, 2, 4, 8, 8], num_res_blocks=1,
                              latents_mean=[0.] * 64, latents_std=[1.] * 64)
model = QwenImage21Transformer2DModel(
    in_channels=64, out_channels=64, num_layers=2, attention_head_dim=32,
    num_attention_heads=8, context_in_dim=32, axes_dims_rope=(8, 12, 12))
scheduler = QwenImage21Scheduler(FlowMatchEulerDiscreteScheduler(
    use_dynamic_shifting=True, max_image_seq_len=8192, max_shift=0.9, shift_terminal=0.02))
conf = OmegaConf.structured(TrainerConfig(
    train_unet=False, train_text_encoder=False, train_dtype="torch.float32",
    weight_dtype="torch.bfloat16", autocast_dtype="torch.bfloat16", vae_dtype="torch.float32",
    lr=1e-3, convrot=args.convrot, gradient_checkpointing=True, te_device=args.te_device, vae_device="cpu",
    validation_args={"prompt": "A red square.", "negative_prompt": ""}))
trainer = BaseTrainer(conf, "qwenimage21", QwenImage21DiffusionModel(model), text, vae,
                      scheduler.diffusers_scheduler, scheduler, None)
trainer.prepare_modules_for_training()
if args.convrot:
    from comfy_kitchen.tensor import QuantizedTensor
    quantized = sum(isinstance(m.weight, QuantizedTensor)
                    for m in model.modules() if isinstance(m, torch.nn.Linear))
    assert quantized > 0, "ConvRot test must actually quantize layers."
    print("ConvRot:", args.convrot, "quantized layers:", quantized, flush=True)
trainer.prepare_network(OmegaConf.create({"module":"networks.manager.NetworkManager", "resume":None,
    "train": True, "args":{"module":"networks.lora.LoRAModule","module_args":{"rank":2,"alpha":2}}}))
trainer.prepare_controlnet(None)
trainer.apply_module_settings()
trainer.prepare_optimizer()
trainer.prepare_lr_scheduler(2)
prompts = ["A red square.", "A small red square on a plain white background."]
with torch.no_grad():
    latents = trainer.encode_latents([Image.new("RGB", (64, 96), "red")] * 2)
batch = {"latents": latents.detach().cpu(), "captions": prompts}
print("Latent shape:", tuple(latents.shape), "Live captions:", len(prompts), flush=True)
initial_lora = {name: param.detach().clone() for name, param in trainer.network.named_parameters()}
for _ in range(2):
    metrics = trainer.step(batch)
    assert np.isfinite(metrics["loss_ema"])
    print("Train:", metrics, flush=True)
assert any(not torch.equal(param.detach(), initial_lora[name])
           for name, param in trainer.network.named_parameters()), "LoRA weights must update."
assert all(torch.isfinite(param).all() for param in trainer.network.parameters())
image = trainer.sample("A red square.", "", height=96, width=64, num_inference_steps=2, guidance_scale=1.)[0]
mode = args.convrot or "bf16"
image.save(f"outputs/qwen21_smoke/tiny_sample_{mode}.png")
assert image.size == (64, 96) and image.mode == "RGBA"
print("GPU pipeline, 2 LoRA steps, RGBA sample: OK", flush=True)

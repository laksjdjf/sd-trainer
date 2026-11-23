"""Offline MOSS integration checks: uv run python -m tools.verify_moss_image."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2
from transformers import BatchEncoding, PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5Model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from modules.dataset import BaseDataset
from modules.config import TrainerConfig
from modules.diffusion.moss_image import MOSSImageDiffusionModel
from modules.model_registry import get_model_spec
from modules.moss_image_latents import patchify, unpatchify
from modules.moss_image_model import MOSSImageTransformer, MOSSImageScheduler
from modules.moss_image_scheduler import MOSSImageTrainingScheduler
from modules.text.base import BaseTextOutput
from modules.text.moss_image import MOSSImageTextModel
from modules.trainer import BaseTrainer
from networks.manager import NetworkManager
from preprocess.latents import _encode_batch
from preprocess.text_emb import encode_text


def tiny_model(layers=2, channels=128):
    return MOSSImageTransformer(dim=32, ffn_dim=64, num_heads=2, num_layers=layers,
                                text_dim=12, freq_dim=16, latent_channels=channels)


def tiny_vae():
    return AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), norm_num_groups=4,
                             layers_per_block=1, mid_block_add_attention=False)


class MOSSImageTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_checkpointing_lora_update_and_reload(self):
        model = tiny_model().requires_grad_(False)
        pristine = copy.deepcopy(model)
        wrapper = MOSSImageDiffusionModel(model)
        text_model = SimpleNamespace(text_encoders=[torch.nn.Identity()])
        network = NetworkManager(text_model, model, module="networks.lora.LoRAModule",
                                 module_args={"rank": 2, "alpha": 2})
        self.assertEqual(len(network.unet_modules), 22)  # 2*6 DiT + 2*5 text-refinement
        x, t = torch.randn(2, 128, 2, 3), torch.tensor([123., 876.])
        text = BaseTextOutput(torch.randn(2, 5, 12), attention_mask=torch.tensor([
            [1, 1, 0, 0, 0], [1, 1, 1, 1, 1],
        ]))
        baseline = wrapper(x, t, text).detach()
        # Padded tokens must not influence text refinement or joint attention.
        altered = text.clone()
        altered.encoder_hidden_states[0, 2:] = 10000
        torch.testing.assert_close(wrapper(x, t, altered), baseline)
        model.enable_gradient_checkpointing()
        torch.testing.assert_close(wrapper(x, t, text), baseline)
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
        loss = (wrapper(x, t, text) - torch.randn_like(x)).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in network.parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        optimizer.step()
        expected = wrapper(x, t, text).detach()
        self.assertFalse(torch.equal(expected, baseline))
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "lora.safetensors")
            network.save_weights(path, dtype=torch.float32)
            NetworkManager(text_model, pristine, file_name=path)
            torch.testing.assert_close(MOSSImageDiffusionModel(pristine)(x, t, text), expected)

    def test_latent_channel_order_normalization_and_cache(self):
        vae = tiny_vae().eval()
        vae.bn.running_mean.copy_(torch.linspace(-1, 1, 128))
        vae.bn.running_var.copy_(torch.linspace(0.5, 2, 128))
        trainer = BaseTrainer(None, "moss_image", None, None, vae, None, None, None)
        image = torch.randn(3, 32, 48)
        torch.manual_seed(17)
        raw = vae.encode(image[None]).latent_dist.sample()
        packed = patchify(raw)
        self.assertEqual(packed.shape, (1, 128, 2, 3))
        torch.testing.assert_close(packed[0, 5 * 4 + 2, 0, 1], raw[0, 5, 1, 2])
        torch.testing.assert_close(unpatchify(packed), raw)
        normalized = trainer.normalize_latents(raw)
        expected = (packed - vae.bn.running_mean.view(1, -1, 1, 1)) / (
            vae.bn.running_var + vae.config.batch_norm_eps).sqrt().view(1, -1, 1, 1)
        torch.testing.assert_close(normalized, expected)
        torch.testing.assert_close(trainer.denormalize_latents(normalized), raw)
        torch.manual_seed(17)
        cached = torch.from_numpy(_encode_batch(vae, [image], False, moss_image=True))
        torch.testing.assert_close(trainer.normalize_latents(cached), normalized)
        self.assertEqual(vae.decode(trainer.denormalize_latents(normalized)).sample.shape, (1, 3, 32, 48))

    def test_text_unpadded_forwards_and_cache_dataset_roundtrip(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return messages[0]["content"]

            def __call__(self, prompt, **kwargs):
                ids = torch.arange(len(prompt) + 1)[None]
                return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.lengths = []

            @property
            def device(self):
                return self.weight.device

            def forward(self, input_ids, attention_mask, use_cache, return_dict):
                self.lengths.append(input_ids.shape[1])
                self.assert_no_cache = not use_cache
                return SimpleNamespace(last_hidden_state=input_ids[..., None].expand(-1, -1, 12).float())

        encoder = Encoder()
        model = MOSSImageTextModel(Tokenizer(), encoder)
        output = model(["a", "abcd"])
        self.assertEqual(encoder.lengths, [2, 5])
        self.assertTrue(encoder.assert_no_cache)
        self.assertEqual(output.attention_mask.sum(1).tolist(), [2, 5])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "captions").mkdir()
            (root / "latents_moss_image").mkdir()
            for name, caption in [("a", "a"), ("b", "abcd")]:
                (root / "captions" / f"{name}.caption").write_text(caption)
                np.save(root / "latents_moss_image" / f"{name}.npy", np.zeros((128, 2, 3), dtype=np.float32))
            (root / "buckets.json").write_text(json.dumps({"48,32": ["a", "b"]}))
            with patch("preprocess.text_emb.load_text_model", return_value=model):
                encode_text(str(root / "captions"), str(root / "text_emb_moss_image"), "unused",
                            model_type="moss_image", save_dtype="fp32")
            dataset = BaseDataset(model, 2, d, model_type="moss_image", latent="latents_moss_image",
                                  text_emb="text_emb_moss_image")
            dataset.batch_samples = [["a", "b"]]
            batch = dataset[0]
            torch.testing.assert_close(batch["encoder_hidden_states"], output.encoder_hidden_states)
            torch.testing.assert_close(batch["attention_mask"], output.attention_mask)
            self.assertEqual(batch["size_condition"][0, -2:].tolist(), [32, 48])
            np.savez(root / "text_emb_moss_image" / "a.npz", encoder_hidden_state=np.zeros((2, 12)))
            with self.assertRaisesRegex(ValueError, "attention_mask"):
                dataset[0]

    def test_scheduler_grid_and_euler(self):
        scheduler = MOSSImageTrainingScheduler(MOSSImageScheduler())
        times = scheduler.set_timesteps(50, device="cpu")
        self.assertEqual(len(times), 50)
        self.assertEqual(times[0].item(), 1000)
        self.assertTrue(torch.all(times[:-1] > times[1:]))
        torch.testing.assert_close(times, times.round())
        self.assertEqual(scheduler.shift, 1.0)
        x, noise = torch.randn(2, 128, 2, 3), torch.randn(2, 128, 2, 3)
        t = torch.tensor([123., 876.])
        noisy = scheduler.add_noise(x, noise, t)
        target = scheduler.get_target(x, noise, t)
        torch.testing.assert_close(scheduler.pred_original_sample(noisy, target, t), x)
        torch.testing.assert_close(scheduler.step(noisy, target, t, t.new_zeros(2)), x)

    def test_local_component_loading_real_qwen35_and_loss(self):
        config = Qwen3_5Config(text_config=dict(
            vocab_size=16, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            layer_types=["linear_attention", "full_attention"],
            linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=8,
            rope_parameters=dict(rope_type="default", rope_theta=10000., partial_rotary_factor=1., mrope_section=[1, 1, 2]),
        ), vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                              out_hidden_size=32, num_position_embeddings=16))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "a": 2}, unk_token="[UNK]")),
            unk_token="[UNK]", pad_token="[PAD]",
            chat_template="{{ messages[0]['content'] }}",
        )
        # MOSS's tokenizer, like Qwen2Tokenizer, does not produce token_type_ids.
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            tokenizer.save_pretrained(root / "tokenizer")
            Qwen3_5Model(config).save_pretrained(root / "text_encoder")
            model = tiny_model()
            # Match the tiny real encoder's text width.
            model = MOSSImageTransformer.from_config(model.config, text_dim=32)
            model.save_pretrained(root / "transformer")
            tiny_vae().save_pretrained(root / "vae")
            MOSSImageScheduler().save_pretrained(root / "scheduler")
            trainer_config = TrainerConfig(train_unet=False, train_text_encoder=False,
                                          weight_dtype="torch.bfloat16")
            trainer = BaseTrainer.from_pretrained(d, "moss_image", torch_dtype=torch.float32,
                                                  config=trainer_config)
            output = trainer.text_model(["a", "a a a"])
            self.assertEqual(output.encoder_hidden_states.shape[-1], 32)
            self.assertTrue(torch.isfinite(output.encoder_hidden_states).all())
            trainer.device = "cuda" if torch.cuda.is_available() else "cpu"
            trainer.autocast_dtype = torch.bfloat16
            trainer.config = SimpleNamespace(noise_offset=0.0)
            trainer.diffusion.to(trainer.device).requires_grad_(False)
            trainer.scaling_factor = trainer.scaling_factor.to(trainer.device)
            trainer.shift_factor = trainer.shift_factor.to(trainer.device)
            network = NetworkManager(trainer.text_model, trainer.diffusion.unet,
                                     module="networks.lora.LoRAModule", module_args={"rank": 2})
            network.to(trainer.device)
            trainer.diffusion.enable_gradient_checkpointing()
            loss = trainer.loss({"latents": torch.randn(2, 128, 2, 3),
                                 "encoder_hidden_states": output.encoder_hidden_states.detach(),
                                 "attention_mask": output.attention_mask})
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in network.parameters()))
            spec = get_model_spec("moss_image")
            with self.assertRaisesRegex(ValueError, "unquantized"):
                spec.load(spec, d, -1, None, None, None, object(), False)

    def test_sampling_bf16_and_checkpoint_save_pretrained(self):
        # 23 blocks include the official negative-branch skip index 22.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = tiny_model(layers=23).to(device=device, dtype=torch.bfloat16).eval()
        text = BaseTextOutput(torch.randn(2, 3, 12, device=device, dtype=torch.bfloat16),
                              attention_mask=torch.ones(2, 3, device=device, dtype=torch.bool))
        scheduler = MOSSImageTrainingScheduler(MOSSImageScheduler())
        trainer = SimpleNamespace(
            diffusion=MOSSImageDiffusionModel(model), scheduler=scheduler,
            autocast_dtype=torch.bfloat16, vae_dtype=torch.float32,
            denormalize_latents=lambda x: x, decode_latents=lambda x: [x],
        )
        initial = torch.randn(1, 128, 2, 3, device=device)
        timesteps = scheduler.set_timesteps(3, device=device)
        from modules.moss_image_sampling import denoising_loop
        result = denoising_loop(trainer, initial, timesteps, text, 3.0, None, False)[0]
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result.dtype, torch.float32)
        reference = initial.clone()
        official = MOSSImageScheduler()
        official.set_timesteps(3, device=device)
        with torch.no_grad():
            for t in official.timesteps:
                positive = model(reference, t, text.encoder_hidden_states[1:], text.attention_mask[1:]).sample.float()
                negative = model(reference, t, text.encoder_hidden_states[:1], text.attention_mask[:1], skip_layers=[22]).sample.float()
                reference = official.step(negative + 3 * (positive - negative), t, reference).prev_sample
        torch.testing.assert_close(result, reference)
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d)
            restored = MOSSImageTransformer.from_pretrained(d, torch_dtype=torch.bfloat16).to(device)
            with torch.no_grad():
                torch.testing.assert_close(restored(initial, timesteps[0], text.encoder_hidden_states[:1]).sample,
                                           model(initial, timesteps[0], text.encoder_hidden_states[:1]).sample)


if __name__ == "__main__":
    unittest.main()

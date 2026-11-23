"""Offline integration checks: python -m tools.verify_qwenimage21 (no model downloads)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from diffusers import QwenImage21Transformer2DModel, FlowMatchEulerDiscreteScheduler
from modules.diffusion.qwenimage21 import QwenImage21DiffusionModel
from modules.text.base import BaseTextOutput
from modules.dataset import BaseDataset
from modules.qwenimage21_scheduler import QwenImage21Scheduler
from modules.trainer import BaseTrainer
from networks.manager import NetworkManager


def tiny_model():
    return QwenImage21Transformer2DModel(
        in_channels=4, out_channels=4, num_layers=2, attention_head_dim=16,
        num_attention_heads=2, context_in_dim=12, axes_dims_rope=(4, 6, 6),
    )


class QwenImage21Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_lora_gradients_checkpointing_and_roundtrip(self):
        model = tiny_model().requires_grad_(False)
        pristine = copy.deepcopy(model)
        model.enable_gradient_checkpointing()
        wrapper = QwenImage21DiffusionModel(model)
        text_model = SimpleNamespace(text_encoders=[torch.nn.Identity()])
        network = NetworkManager(text_model, model, module="networks.lora.LoRAModule",
                                 module_args={"rank": 2, "alpha": 2})
        self.assertGreater(len(network.unet_modules), 0)
        x = torch.randn(2, 4, 1, 4, 6)
        text = BaseTextOutput(torch.randn(2, 5, 12),
                              attention_mask=torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]))
        t = torch.tensor([100., 800.])
        baseline = wrapper(x, t, text).detach()
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
            restored = NetworkManager(text_model, pristine, file_name=path)
            actual = QwenImage21DiffusionModel(pristine)(x, t, text)
            torch.testing.assert_close(actual, expected)
        # Changing masked prompt values must not affect generated image tokens.
        changed = text.clone()
        changed.encoder_hidden_states[0, 3:] = 1000
        torch.testing.assert_close(wrapper(x, t, changed), expected, atol=1e-5, rtol=1e-5)

    def test_cache_padding_dropout_rgba_and_size(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for sub in ("images", "latents", "text_emb"):
                (root / sub).mkdir()
            (root / "buckets.json").write_text(json.dumps({"64x96": ["a", "b"]}))
            for name, length in [("a", 2), ("b", 5)]:
                Image.new("RGB", (96, 64), "red").save(root / "images" / (name + ".png"))
                np.save(root / "latents" / (name + ".npy"), np.zeros((64, 1, 4, 6), np.float32))
                np.savez(root / "text_emb" / (name + ".npz"),
                         encoder_hidden_state=np.ones((length, 12), np.float32),
                         attention_mask=np.ones(length, bool))
            uncond = BaseTextOutput(torch.zeros(1, 1, 12), attention_mask=torch.ones(1, 1, dtype=torch.bool))
            dataset = BaseDataset(SimpleNamespace(uncond_output=uncond), 2, d,
                                  model_type="qwenimage21", text_emb="text_emb")
            batch = dataset[0]
            self.assertEqual(batch["encoder_hidden_states"].shape, (2, 5, 12))
            self.assertEqual(sorted(batch["attention_mask"].sum(1).tolist()), [2, 5])
            self.assertEqual(batch["size_condition"][0, -2:].tolist(), [64, 96])
            self.assertNotIn("pooled_outputs", batch)
            dataset.ucg = 1
            self.assertEqual(dataset[0]["encoder_hidden_states"].shape, (2, 1, 12))
            dataset.image = "images"
            self.assertEqual(dataset[0]["images"].shape, (2, 4, 64, 96))
            self.assertTrue((dataset[0]["images"][:, 3] == 1).all())

    def test_text_preprocessor_writes_mask_without_pooling(self):
        from preprocess.text_emb import encode_text
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            captions = root / "captions"
            captions.mkdir()
            for name in ("a", "b"):
                (captions / (name + ".caption")).write_text(name)
            output = BaseTextOutput(torch.randn(2, 5, 12),
                                    attention_mask=torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]]))
            with patch("preprocess.text_emb.load_text_model", return_value=lambda texts: output):
                encode_text(str(captions), str(root / "text_emb"), "unused", model_type="qwenimage21")
            with np.load(root / "text_emb/a.npz") as cache:
                self.assertNotIn("pooled_output", cache)
                self.assertEqual(cache["attention_mask"].dtype, np.bool_)
                self.assertEqual(cache["attention_mask"].sum(), 2)
                self.assertEqual(cache["encoder_hidden_state"].dtype, np.float16)

    def test_legacy_pooled_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "buckets.json").write_text('{"64x64":["a"]}')
            (root / "text_emb").mkdir()
            np.savez(root / "text_emb/a.npz", encoder_hidden_state=np.zeros((3, 8)),
                     pooled_output=np.ones(4))
            dataset = BaseDataset(None, 1, d)
            hidden, pooled = dataset.get_text_embeddings(["a"])
            self.assertEqual(hidden.shape, (1, 3, 8))
            self.assertEqual(pooled.shape, (1, 4))

    def test_official_sampling_schedule_and_latent_shape(self):
        scheduler = QwenImage21Scheduler(FlowMatchEulerDiscreteScheduler(
            use_dynamic_shifting=True, max_image_seq_len=8192, max_shift=0.9,
            shift_terminal=0.02,
        ))
        for h, w in [(64, 64), (64, 96)]:
            scheduler.configure_latents((1, 64, 1, h, w))
            timesteps = scheduler.set_timesteps(4, "cpu")
            x, prediction = torch.randn(1, 4, 1, h, w), torch.randn(1, 4, 1, h, w)
            ref = scheduler.diffusers_scheduler
            for i, t in enumerate(timesteps):
                expected = ref.step(prediction, t, x, return_dict=False)[0]
                actual = (scheduler.step(x, prediction, t, timesteps[i+1])
                          if i+1 < len(timesteps) else scheduler.pred_original_sample(x, prediction, t))
                torch.testing.assert_close(actual, expected)
                x = actual
        vae = SimpleNamespace(config=SimpleNamespace(latents_mean=[0.] * 64, latents_std=[1.] * 64))
        trainer = BaseTrainer(None, "qwenimage21", None, None, vae, None, scheduler, None)
        trainer.device, trainer.autocast_dtype = "cpu", torch.float32
        self.assertEqual(trainer._init_sample_latents(2, 1024, 1536, None, None).shape, (2, 64, 1, 64, 96))
        with self.assertRaises(ValueError):
            trainer._init_sample_latents(1, 1000, 1024, None, None)


if __name__ == "__main__":
    unittest.main()

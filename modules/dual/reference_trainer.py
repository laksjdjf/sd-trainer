import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.trainer import BaseTrainer
from modules.text_model import BaseTextOutput
from diffusers.models.attention_processor import AttnProcessor2_0

class ReferenceAttentionProcessor(AttnProcessor2_0):
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if batch_size > 1:
            key_ref = key[:1].repeat(batch_size, 1, 1, 1)
            value_ref = value[:1].repeat(batch_size, 1, 1, 1)

            key = torch.cat([key_ref, key], dim=2)
            value = torch.cat([value_ref, value], dim=2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class ReferenceIdentityTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion.unet.set_attn_processor(ReferenceAttentionProcessor())

    def loss(self, batch):
        if not hasattr(self, "alternate"):
            self.alternate = 1.0

        if "latents_w" in batch:
            latents_w = batch["latents_w"].to(self.device)
            latents_l = batch["latents_l"].to(self.device)
        else:
            with torch.autocast("cuda", dtype=self.vae_dtype), torch.no_grad():
                latents_w = self.vae.encode(batch['images_w'].to(self.device)).latent_dist.sample()
                latents_l = self.vae.encode(batch['images_l'].to(self.device)).latent_dist.sample()
        latents_w = (latents_w - self.shift_factor) * self.scaling_factor
        latents_l = (latents_l - self.shift_factor) * self.scaling_factor

        self.batch_size = latents_w.shape[0] # stepメソッドでも使う

        if "encoder_hidden_states" in batch:
            encoder_hidden_states = batch["encoder_hidden_states"].to(self.device)
            pooled_output = batch["pooled_outputs"].to(self.device)
            text_output_w = BaseTextOutput(encoder_hidden_states, pooled_output)
        else:
            with torch.autocast("cuda", dtype=self.autocast_dtype):
                text_output_w = self.text_model(batch["captions"])

        if "encoder_hidden_states_l" in batch:
            encoder_hidden_states_l = batch["encoder_hidden_states_l"].to(self.device)
            pooled_output_l = batch["pooled_outputs_l"].to(self.device)
            text_output_l = BaseTextOutput(encoder_hidden_states_l, pooled_output_l)
        else:
            if "captions_l" in batch:
                with torch.autocast("cuda", dtype=self.autocast_dtype):
                    text_output_l = self.text_model(batch["captions_l"])
            else:
                text_output_l = text_output_w

        if self.alternate == -1.0:
            latents_w, latents_l = latents_l, latents_w
            text_output_w, text_output_l = text_output_l, text_output_w

        text_output = BaseTextOutput.cat([text_output_w, text_output_l])

        timesteps = self.scheduler.sample_timesteps(latents_w.shape[0], self.device)
        noise = torch.randn_like(latents_w)
        if self.config.noise_offset != 0:
            noise += self.config.noise_offset * torch.randn(noise.shape[0], noise.shape[1], 1, 1).to(noise)
        noisy_latents_w = self.scheduler.add_noise(latents_w, noise, timesteps)
        noisy_latents_l = self.scheduler.add_noise(latents_l, noise, timesteps)
        noisy_latents = torch.cat([noisy_latents_w, noisy_latents_l], dim=0)
        
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            model_output = self.diffusion(noisy_latents, timesteps.repeat(2), text_output, sample=False)[1:]
        target = self.scheduler.get_target(latents_w, noise, timesteps)
        loss = nn.functional.mse_loss(model_output.float(), target.float(), reduction="mean")

        self.alternate *= -1
        
        return loss
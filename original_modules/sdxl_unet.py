# ref:https://github.com/cloneofsimo/minSDXL

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from collections import namedtuple

# SDXL

@dataclass
class SampleOutput:
    sample: torch.FloatTensor = None

class Timesteps(nn.Module):
    def __init__(self, num_channels: int = 320):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(
            half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features, out_features):
        super(TimestepEmbedding, self).__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, out_features, bias=True)

    def forward(self, sample):

        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)

        return sample


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResnetBlock2D, self).__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-05, affine=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.time_emb_proj = nn.Linear(1280, out_channels, bias=True)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-05, affine=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.nonlinearity = nn.SiLU()
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1
            )
        else:
            self.conv_shortcut = None

    def forward(self, x, temb):
        hidden_states = self.norm1(x)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]
        hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)

        output_tensor = x + hidden_states

        return output_tensor


class Attention(nn.Module):
    def __init__(self, inner_dim, is_cross=False):
        super(Attention, self).__init__()
        self.head_dim = 64
        self.num_heads = inner_dim // self.head_dim

        self.scale = self.head_dim**-0.5
        cross_attention_dim = 2048 if is_cross else inner_dim
        self.to_q = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, inner_dim), nn.Identity()]  # dropout but default is 0
        )

    def forward(self, hidden_states, encoder_hidden_states=None):
        kv_inputs = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        q = self.to_q(hidden_states)
        k = self.to_k(kv_inputs)
        v = self.to_v(kv_inputs)
        b, t, c = q.size()

        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, c)

        attn_output = self.to_out[0](attn_output)  # to_out[1] is identity
        return attn_output


class GEGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super(GEGLU, self).__init__()
        self.proj = nn.Linear(in_features, out_features * 2, bias=True)

    def forward(self, x):
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return x1 * torch.nn.functional.gelu(x2)


class FeedForward(nn.Module):
    def __init__(self, in_features, out_features):
        super(FeedForward, self).__init__()

        self.net = nn.ModuleList(
            [
                GEGLU(in_features, out_features * 4),
                nn.Dropout(p=0.0, inplace=False),
                nn.Linear(out_features * 4, out_features, bias=True),
            ]
        )

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(self, hidden_size):
        super(BasicTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.attn1 = Attention(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.attn2 = Attention(hidden_size, is_cross=True)
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.ff = FeedForward(hidden_size, hidden_size)

    def forward(self, x, encoder_hidden_states):

        attn1_output = self.norm1(x)
        attn1_output = self.attn1(attn1_output) + x

        attn2_output = self.norm2(attn1_output)
        attn2_output = self.attn2(attn2_output, encoder_hidden_states) + attn1_output

        ff_output = self.norm3(attn2_output)
        ff_output = self.ff(ff_output) + attn2_output
        return ff_output


class Transformer2DModel(nn.Module):
    def __init__(self, channels, n_layers):
        super(Transformer2DModel, self).__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-06, affine=True)
        self.proj_in = nn.Linear(channels, channels, bias=True)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(channels) for _ in range(n_layers)]
        )
        self.proj_out = nn.Linear(channels, channels, bias=True)

    def forward(self, x, encoder_hidden_states):
        batch, _, height, width = x.shape
        hidden_states = self.norm(x)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, height * width, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states)

        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, width, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return hidden_states + x


class Downsample2D(nn.Module):
    def __init__(self, channels):
        super(Downsample2D, self).__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, channels):
        super(Upsample2D, self).__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = x.to(dtype)
        return self.conv(x)


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, prev_channels, block, transformer_depth=0, final=False):
        super(UNetBlock, self).__init__()
        self.up = block == "up"
        self.down = block == "down"
        self.mid = block == "mid"
        self.final = final
        self.transformer_depth = transformer_depth

        self.layers = 3 if self.up else 2
        if self.up:
            in_channels_list = [
                in_channels + out_channels,
                out_channels + out_channels,
                out_channels + prev_channels,
            ]
        else:
            in_channels_list = [in_channels] + [out_channels]

        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels_list[i], out_channels)
            for i in range(self.layers)
        ])

        if transformer_depth > 0:
            self.attentions = nn.ModuleList([
                Transformer2DModel(out_channels, self.transformer_depth)
                for _ in range(1 if self.mid else self.layers)
            ])
        else:
            self.attentions = []

        if not self.final:
            if self.up:
                self.upsamplers = nn.ModuleList([Upsample2D(out_channels)])
            if self.down:
                self.downsamplers = nn.ModuleList([Downsample2D(out_channels)])

    def forward(self, hidden_states, res_samples, encoder_hidden_states, temb):
        res_output = ()
        for i in range(self.layers):
            if self.up:
                hidden_states = torch.cat([hidden_states, res_samples[-(i+1)]], dim=1)

            hidden_states = self.resnets[i](hidden_states, temb)
            if len(self.attentions) > i:
                hidden_states = self.attentions[i](hidden_states, encoder_hidden_states)

            if self.down:
                res_output = res_output + (hidden_states,)

        if not self.final:
            if self.up:
                hidden_states = self.upsamplers[0](hidden_states)

            if self.down:
                hidden_states = self.downsamplers[0](hidden_states)
                res_output = res_output + (hidden_states,)

        return hidden_states, res_output


class UNet2DConditionModel(nn.Module):
    def __init__(self):
        super(UNet2DConditionModel, self).__init__()

        self.config = namedtuple("config", "in_channels addition_time_embed_dim")
        self.config.in_channels = 4
        self.config.addition_time_embed_dim = 256

        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, stride=1, padding=1)
        self.time_proj = Timesteps()
        self.time_embedding = TimestepEmbedding(in_features=320, out_features=1280)
        self.add_time_proj = Timesteps(256)
        self.add_embedding = TimestepEmbedding(in_features=2816, out_features=1280)

        self.down_blocks = nn.ModuleList([
            UNetBlock(320, 320, None, block="down"),
            UNetBlock(320, 640, None, block="down", transformer_depth=2),
            UNetBlock(640, 1280, None, block="down", transformer_depth=10, final=True),
        ])

        self.mid_block = UNetBlock(1280, 1280, None, block="mid", transformer_depth=10)

        self.up_blocks = nn.ModuleList([
            UNetBlock(1280, 1280, 640, block="up", transformer_depth=10),
            UNetBlock(1280, 640, 320, block="up", transformer_depth=2),
            UNetBlock(640, 320, 320, block="up", final=True),
        ])

        self.conv_norm_out = nn.GroupNorm(32, 320, eps=1e-05, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, stride=1, padding=1)

    def forward(
        self, sample, timesteps, encoder_hidden_states, added_cond_kwargs, return_dict=True, **kwargs
    ):
        # Implement the forward pass through the model
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        # print(t_emb)
        emb = self.time_embedding(t_emb)

        text_embeds = added_cond_kwargs.get("text_embeds")
        time_ids = added_cond_kwargs.get("time_ids")

        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1).to(emb.dtype)
        aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb

        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if downsample_block.transformer_depth > 0:
                encoder_hidden_states_input = encoder_hidden_states
            else:
                encoder_hidden_states_input = None

            sample, res_samples = downsample_block(sample, None, encoder_hidden_states_input, emb)
            down_block_res_samples += res_samples

        # 4. mid
        sample, _ = self.mid_block(sample, None, encoder_hidden_states, emb)

        # 5. up
        for upsample_block in self.up_blocks:
            if upsample_block.transformer_depth > 0:
                encoder_hidden_states_input = encoder_hidden_states
            else:
                encoder_hidden_states_input = None

            res_samples = down_block_res_samples[-3:]
            down_block_res_samples = down_block_res_samples[:-3]
            sample, _ = upsample_block(sample, res_samples, encoder_hidden_states_input, emb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)
        else:
            return SampleOutput(sample=sample)

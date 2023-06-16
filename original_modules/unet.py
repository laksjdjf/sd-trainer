#Kohyaさんのoriginal unetをさらに簡略化する。

# Kohya's memo
# 条件分岐等で不要な部分は削除している
# コードの多くはDiffusersからコピーしている
# 制約として、モデルのstate_dictがDiffusers 0.10.2のものと同じ形式である必要がある
# Copy from Diffusers 0.10.2 for Stable Diffusion. Most of the code is copied from Diffusers.
# Unnecessary parts are deleted by condition branching.
# As a constraint, the state_dict of the model must be in the same format as that of Diffusers 0.10.2

# 以下私のメモ
# diffusers 0.16.1までなら変わらないと思う
# 方針として拡張性・柔軟性を犠牲にして可読性を上げている（はず）。
# 冗長なモジュールをカット
# ループを使った複雑な定義を簡略化->他の構造には対応できない。
# 変数の伝言ゲームを廃止・グローバル変数に頼る->二つのUNetを同時に扱うとき困るが、いやそんなことしないだろ。

from typing import Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from types import SimpleNamespace

BLOCK_OUT_CHANNELS: Tuple[int] = (320, 640, 1280, 1280)
TIMESTEP_INPUT_DIM = BLOCK_OUT_CHANNELS[0]
TIME_EMBED_DIM = BLOCK_OUT_CHANNELS[0] * 4
IN_CHANNELS: int = 4
OUT_CHANNELS: int = IN_CHANNELS
LAYERS_PER_BLOCK: int = 2
LAYERS_PER_BLOCK_UP: int = LAYERS_PER_BLOCK + 1
TIME_EMBED_FLIP_SIN_TO_COS: bool = True
TIME_EMBED_FREQ_SHIFT: int = 0
NORM_GROUPS: int = 32
NORM_EPS: float = 1e-5
TRANSFORMER_NORM_NUM_GROUPS = 32

class V1Config:
    attention_heads: int = 8
    attention_head_dims: int = None
    upcast_attention: bool = False
    use_linear_projection: bool = False
    cross_attention_dim: int = 768

class V2Config:
    attention_heads: int = None
    attention_head_dims: int = 64
    upcast_attention: bool = True
    use_linear_projection: bool = True
    cross_attention_dim: int = 1024

class GlobalConfig:
    attention_mode:str = None
    gradient_checkpointing: bool = False

class SampleOutput:
    def __init__(self, sample):
        self.sample = sample

def get_parameter_dtype(parameter: torch.nn.Module):
    return next(parameter.parameters()).dtype

def get_parameter_device(parameter: torch.nn.Module):
    return next(parameter.parameters()).device

class Timesteps(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, timesteps):
        """
        This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
        embeddings. :return: an [N x dim] Tensor of positional embeddings.
        """
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

        half_dim = TIMESTEP_INPUT_DIM // 2
        exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / half_dim

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        # concat sine and cosine embeddings
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # flip sine and cosine embeddings
        if TIME_EMBED_FLIP_SIN_TO_COS:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        # zero pad
        if TIMESTEP_INPUT_DIM % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

class TimestepEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(TIMESTEP_INPUT_DIM, TIME_EMBED_DIM)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(TIME_EMBED_DIM, TIME_EMBED_DIM)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample

class ResnetBlock2D(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = torch.nn.GroupNorm(num_groups=NORM_GROUPS, num_channels=in_channels, eps=NORM_EPS, affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.time_emb_proj = torch.nn.Linear(TIME_EMBED_DIM, out_channels)
        self.norm2 = torch.nn.GroupNorm(num_groups=NORM_GROUPS, num_channels=out_channels, eps=NORM_EPS, affine=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # if non_linearity == "swish":
        self.nonlinearity = lambda x: F.silu(x)

        self.use_in_shortcut = self.in_channels != self.out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
        hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor
    
class Downsample2D(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, stride=2, padding=1)

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states
    
class CrossAttention(nn.Module):
    def __init__(self, dim: int, is_cross:bool):
        super().__init__()
        if version_config.attention_heads is not None:
            self.heads = version_config.attention_heads
            dim_head = dim // self.heads
        else:
            dim_head = version_config.attention_head_dims
            self.heads = dim // dim_head
        cross_attention_dim = version_config.cross_attention_dim if is_cross else dim
        self.upcast_attention = version_config.upcast_attention

        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, dim, bias=False)

        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])
        # no dropout here

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def forward(self, hidden_states, context=None, mask=None):
        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        hidden_states = self._attention(query, key, value, mask)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # hidden_states = self.to_out[1](hidden_states)     # no dropout
        return hidden_states
    
    def _attention(self, query, key, value, mask):
        if global_config.attention_mode == "xformers":
            return self._attention_xformers(query, key, value, mask)
        if global_config.attention_mode == "sdpa":
            return F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False)
        else:
            return self._attention_normal(query, key, value, mask)

    def _attention_normal(self, query, key, value, mask=None):
        if version_config.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)
        
        return hidden_states

    def _attention_xformers(self, query, key, value, mask=None):
        import xformers.ops
        q = query.contiguous()
        k = key.contiguous()
        v = value.contiguous()
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
        return out
    
# feedforward
class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inner_dim = int(dim * 4)  # mult is always 4

        self.net = nn.ModuleList([
            GEGLU(dim, inner_dim),
            nn.Identity(),  # nn.Dropout(0), # dummy for dropout with 0
            nn.Linear(inner_dim, dim)
        ])

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
    
class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        # 1. Self-Attn
        self.attn1 = CrossAttention(dim=dim, is_cross=False)
        self.ff = FeedForward(dim)

        # 2. Cross-Attn
        self.attn2 = CrossAttention(dim=dim, is_cross=True)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, hidden_states, context=None, timestep=None):
        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)

        hidden_states = self.attn1(norm_hidden_states) + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        hidden_states = self.attn2(norm_hidden_states, context=context) + hidden_states

        # 3. Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states
    
class Transformer2DModel(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = torch.nn.GroupNorm(num_groups=TRANSFORMER_NORM_NUM_GROUPS, num_channels=dim, eps=1e-6, affine=True)

        if version_config.use_linear_projection:
            self.proj_in = nn.Linear(dim, dim)
        else:
            self.proj_in = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(dim)])

        if version_config.use_linear_projection:
            self.proj_out = nn.Linear(dim, dim)
        else:
            self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True):
        # 1. Input
        batch, _, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not version_config.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, context=encoder_hidden_states, timestep=timestep)

        # 3. Output
        if not version_config.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return SampleOutput(sample=output)

class CrossAttnDownBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, final_block: bool = False):
        super().__init__()
        self.final_block = final_block
        
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels=in_channels if i == 0 else out_channels, out_channels=out_channels) 
            for i in range(LAYERS_PER_BLOCK)
        ])
        
        if not self.final_block:
            self.attentions = nn.ModuleList([Transformer2DModel(out_channels) for _ in range(LAYERS_PER_BLOCK)])
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, out_channels)])
        else:
            self.attentions = [None] * LAYERS_PER_BLOCK # とりあえずNoneのリストにしておく

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            if global_config.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                if not self.final_block:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn, return_dict=False), hidden_states, encoder_hidden_states
                    )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                if not self.final_block:
                    hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample

            output_states += (hidden_states,)
        if not self.final_block:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states
    
class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(self,in_channels: int):
        super().__init__()

        # Middle block has two resnets and one attention
        resnets = [
            ResnetBlock2D(in_channels=in_channels,out_channels=in_channels),
            ResnetBlock2D(in_channels=in_channels,out_channels=in_channels)
        ]
        attentions = [Transformer2DModel(dim=in_channels)]

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        for i, resnet in enumerate(self.resnets):
            attn = None if i == 0 else self.attentions[i - 1]

            if global_config.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                if attn is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn, return_dict=False), hidden_states, encoder_hidden_states
                    )[0]

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                if attn is not None:
                    hidden_states = attn(hidden_states, encoder_hidden_states).sample
                hidden_states = resnet(hidden_states, temb)

        return hidden_states
    
class Upsample2D(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, hidden_states, output_size):
        assert hidden_states.shape[1] == self.channels

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        hidden_states = self.conv(hidden_states)

        return hidden_states
    
class CrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        has_transformers: bool = True,
        final_block: bool = False,
    ):
        super().__init__()
        resnets = []
        attentions = []
        self.has_transformers = has_transformers
        self.final_block = final_block

        in_channels_list = [out_channels * 2] * LAYERS_PER_BLOCK_UP
        in_channels_list[0] = prev_output_channel + out_channels
        in_channels_list[-1] = out_channels + in_channels

        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels=in_channels_list[i], out_channels=out_channels)
            for i in range(LAYERS_PER_BLOCK_UP)
        ])

        if self.has_transformers:
            self.attentions = nn.ModuleList([Transformer2DModel(dim=out_channels) for _ in range(LAYERS_PER_BLOCK_UP)])
        else:
            self.attentions = [None] * LAYERS_PER_BLOCK_UP

        if not self.final_block:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels)])

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if global_config.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                if self.has_transformers:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn, return_dict=False), hidden_states, encoder_hidden_states
                    )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                if self.has_transformers:
                    hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample

        if not self.final_block:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states

class UNet2DConditionModel(nn.Module):

    def __init__(self, version="v1"):
        super().__init__()
        global version_config, global_config
        if version == "v1":
            version_config = V1Config()
        elif version == "v2":
            version_config = V2Config()

        global_config = GlobalConfig()
        
        # 外部からの参照用に定義しておく
        self.in_channels = IN_CHANNELS
        self.out_channels = OUT_CHANNELS
        self.sample_size = 64
        self.prepare_config()

        # state_dictの書式が変わるのでmoduleの持ち方は変えられない

        # input
        self.conv_in = nn.Conv2d(IN_CHANNELS, BLOCK_OUT_CHANNELS[0], kernel_size=3, padding=(1, 1))

        # time
        self.time_proj = Timesteps()

        self.time_embedding = TimestepEmbedding()

        self.down_blocks = nn.ModuleList([
            CrossAttnDownBlock2D(BLOCK_OUT_CHANNELS[0], BLOCK_OUT_CHANNELS[0]),
            CrossAttnDownBlock2D(BLOCK_OUT_CHANNELS[0], BLOCK_OUT_CHANNELS[1]),
            CrossAttnDownBlock2D(BLOCK_OUT_CHANNELS[1], BLOCK_OUT_CHANNELS[2]),
            CrossAttnDownBlock2D(BLOCK_OUT_CHANNELS[2], BLOCK_OUT_CHANNELS[3], final_block=True),
        ])
        self.mid_block = UNetMidBlock2DCrossAttn(in_channels=BLOCK_OUT_CHANNELS[-1])
        self.up_blocks = nn.ModuleList([
            CrossAttnUpBlock2D(BLOCK_OUT_CHANNELS[2], BLOCK_OUT_CHANNELS[3], BLOCK_OUT_CHANNELS[3], has_transformers=False),
            CrossAttnUpBlock2D(BLOCK_OUT_CHANNELS[1], BLOCK_OUT_CHANNELS[2], BLOCK_OUT_CHANNELS[3]),
            CrossAttnUpBlock2D(BLOCK_OUT_CHANNELS[0], BLOCK_OUT_CHANNELS[1], BLOCK_OUT_CHANNELS[2]),
            CrossAttnUpBlock2D(BLOCK_OUT_CHANNELS[0], BLOCK_OUT_CHANNELS[0], BLOCK_OUT_CHANNELS[1], final_block=True),
        ])

        # count how many layers upsample the images
        self.num_upsamplers = 3

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=BLOCK_OUT_CHANNELS[0], num_groups=NORM_GROUPS, eps=NORM_EPS)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(BLOCK_OUT_CHANNELS[0], OUT_CHANNELS, kernel_size=3, padding=1)

    # region diffusers compatibility
    def prepare_config(self):
        self.config = SimpleNamespace()
        self.config.in_channels = self.in_channels
        self.config.out_channels = self.out_channels
        self.config.sample_size = self.sample_size

    @property
    def dtype(self) -> torch.dtype:
        # `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        return get_parameter_dtype(self)

    @property
    def device(self) -> torch.device:
        # `torch.device`: The device on which the module is (assuming that all the module parameters are on the same device).
        return get_parameter_device(self)

    # endregion

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        cross_attention_kwargs=None,
        return_dict: bool = True,
    ) -> Union[Dict, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a dict instead of a plain tuple.

        Returns:
            `SampleOutput` or `tuple`:
            `SampleOutput` if `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        # デフォルトではサンプルは「2^アップサンプルの数」、つまり64の倍数である必要がある
        # ただそれ以外のサイズにも対応できるように、必要ならアップサンプルのサイズを変更する
        # 多分画質が悪くなるので、64で割り切れるようにしておくのが良い
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        # 64で割り切れないときはupsamplerにサイズを伝える
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 1. time
        timesteps = timestep
        timesteps = self.handle_unusual_timesteps(sample, timesteps)  # 変な時だけ処理

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        # timestepsは重みを含まないので常にfloat32のテンソルを返す
        # しかしtime_embeddingはfp16で動いているかもしれないので、ここでキャストする必要がある
        # time_projでキャストしておけばいいんじゃね？
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
            )

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]  # skip connection

            # if we have not reached the final block and need to forward the upsample size, we do it here
            # 前述のように最後のブロック以外ではupsample_sizeを伝える
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                upsample_size=upsample_size,
            )


        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return SampleOutput(sample=sample)
    
    def set_attention_mode(self, mode:str = None):
        if mode == "xformers":
            try:
                import xformers.ops
            except:
                print("can't import xformers")
                return
        if mode == "sdpa":
            if not hasattr(F, "scaled_dot_product_attention"):
                print("This pytorch version dont support sdpa")
                return
        print(f"Set attention_mode to {mode}")
        global_config.attention_mode = mode

    def set_gradient_checkpointing(self, enable=True):
        global_config.gradient_checkpointing = enable

    def handle_unusual_timesteps(self, sample, timesteps):
        r"""
        timestampsがTensorでない場合、Tensorに変換する。またOnnx/Core MLと互換性のあるようにbatchサイズまでbroadcastする。
        """
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        return timesteps
    
    # ださいのでどうにかしたい
    def from_pretrained(model, subfolder):
        from diffusers import UNet2DConditionModel as OriginalUNet
        unet_org = OriginalUNet.from_pretrained(model, subfolder=subfolder)
        if unet_org.config.cross_attention_dim == 768:
            version = "v1"
        else:
            version = "v2"
        unet = UNet2DConditionModel(version)
        unet.load_state_dict(unet_org.state_dict())
        del unet_org
        return unet

if __name__ == "__main__":
    device = "cuda"
    unet = UNet2DConditionModel("v2").to(device)
    latent = torch.randn(1,4,64,64).to(device)
    cond = torch.randn(1,77,1024).to(device)
    timestep = torch.randint(0, 1000, (1,)).to(device)
    output = unet(latent,timestep,cond)
    print(output.sample.shape)

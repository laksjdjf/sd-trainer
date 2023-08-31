import torch
from torch import nn
# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
import math

# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )
    
    
def reshape_tensor(x, heads):
    bs, length, width = x.shape
    #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)
        
        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
    ):
        super().__init__()
        
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        
        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        
        latents = self.latents.repeat(x.size(0), 1, 1)
        
        x = self.proj_in(x)
        
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
            
        latents = self.proj_out(latents)
        return self.norm_out(latents)
    

class IPAttnProcessor(nn.Module):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        text_context_len (`int`, defaults to 77):
            The context length of the text features.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, hidden_size, cross_attention_dim, ip_layer, text_context_len=77, scale=1.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.text_context_len = text_context_len
        self.scale = scale

        self.to_kv = [ip_layer]

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
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
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # for ip-adapter
        ip_key, ip_value = self.to_kv[0].kv

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)
        
        ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)
        
        hidden_states = hidden_states + self.scale * ip_hidden_states

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

class ImageProjModel(nn.Module):
    """Projection Model"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens
    
class To_KV(nn.Module):
    def __init__(self, unet, name, hidden_size, cross_attention_dim=768):
        super().__init__()
        self.to_k_ip  = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip  = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_k_ip.weight.data = unet.state_dict()[name + ".to_k.weight"].clone().detach()
        self.to_v_ip.weight.data = unet.state_dict()[name + ".to_v.weight"].clone().detach()
        self.kv = None
        
    def forward(self, context):
        key = self.to_k_ip(context)
        value = self.to_v_ip(context)
        return key, value
    
    def set_kv(self, context):
        self.kv = self(context)
    
class IPAdapter(nn.Module):    
    def __init__(
        self, 
        unet,
        clip_embeddings_dim=1024,
        clip_extra_context_tokens=4, 
        plus = False,
        resume = None
    ):
        super().__init__()
        self.plus = plus
        attn_procs = {}
        ip_layers = []
        cross_attention_dim = unet.config.cross_attention_dim
        for name in unet.attn_processors.keys():
            attn2 = name.endswith("attn2.processor")
            name_without_suffix = name.replace(".processor", "")
            if attn2:
                hidden_size = unet.state_dict()[name_without_suffix + ".to_q.weight"].shape[0]
                ip_layers.append(To_KV(unet, name_without_suffix, hidden_size, cross_attention_dim=cross_attention_dim))
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    ip_layer=ip_layers[-1],
                )
            else:
                ip_layers.append(nn.Identity())
                attn_procs[name] = unet.attn_processors[name]
        unet.set_attn_processor(attn_procs)
        
        self.ip_layers = nn.ModuleList(ip_layers)

        if plus:
            # image proj model
            self.image_proj_model = Resampler(
                dim=cross_attention_dim,
                depth=4,
                dim_head=64,
                heads=12,
                num_queries=clip_extra_context_tokens,
                embedding_dim=clip_embeddings_dim,
                output_dim=cross_attention_dim,
                ff_mult=4
            )
        else:
            # image proj model
            self.image_proj_model = ImageProjModel(
                cross_attention_dim=cross_attention_dim, 
                clip_embeddings_dim=clip_embeddings_dim, 
                clip_extra_context_tokens=clip_extra_context_tokens
            )
        
        if resume is not None:
            self.load_ip_adapter(resume)
        
    def load_ip_adapter(self, resume):
        state_dict = torch.load(resume, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.ip_layers.load_state_dict(state_dict["ip_adapter"])
        print("学習済みIP-Adapterをロードしました。")

    def save_ip_adapter(self, save_path):
        state_dict = {
            "image_proj": self.image_proj_model.state_dict(),
            "ip_adapter": self.ip_layers.state_dict()
        }
        torch.save(state_dict, save_path)
    
    def trainable_params(self):
        params = []
        params += list(self.image_proj_model.parameters())
        params += list(self.ip_layers.parameters())
        return params
        
    def get_image_embeds(self, clip_image_embeds):
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        return image_prompt_embeds
    
    def set_ip_hidden_states(self, ip_hidden_states):
        for layer in self.ip_layers:
            if isinstance(layer, To_KV):
                layer.set_kv(ip_hidden_states)
    
    def set_scale(self, unet, scale):
        for key in unet.attn_processors.keys():
            if "attn2" in key:
                unet.attn_processors[key].scale = scale
    
    def clip_vision_encode(self, clip_vision, images):
        outputs = clip_vision(images, output_hidden_states=True)
        if self.plus:
            cond = outputs.hidden_states[-2]
            uncond = clip_vision(torch.zeros_like(images), output_hidden_states=True).hidden_states[-2]
        else:
            cond = outputs.image_embeds
            uncond = torch.zeros_like(cond)
        return cond, uncond
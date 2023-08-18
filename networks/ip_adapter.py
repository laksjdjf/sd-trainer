import torch
from torch import nn

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
            
        # split hidden states
        encoder_hidden_states, ip_hidden_states = encoder_hidden_states[:, :self.text_context_len, :], encoder_hidden_states[:, self.text_context_len:, :]

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # for ip-adapter
        ip_key, ip_value = self.to_kv[0](ip_hidden_states)

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
        self.to_k_ip.weight.data = unet.state_dict()[name + ".to_k.weight"].detach()
        self.to_v_ip.weight.data = unet.state_dict()[name + ".to_v.weight"].detach()

    def forward(self, context):
        key = self.to_k_ip(context)
        value = self.to_v_ip(context)
        return key, value
    
class IPAdapter(nn.Module):    
    def __init__(
        self, 
        unet,
        clip_embeddings_dim=1024,
        clip_extra_context_tokens=4, 
        resume = None
    ):
        super().__init__()

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
                    scale=1.0,
                    ip_layer=ip_layers[-1]
                )
            else:
                ip_layers.append(nn.Identity())
                attn_procs[name] = unet.attn_processors[name]
                
        unet.set_attn_processor(attn_procs)
        self.ip_layers = nn.ModuleList(ip_layers)

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
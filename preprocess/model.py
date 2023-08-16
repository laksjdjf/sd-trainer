import torch
import torch.nn as nn
import torch.nn.functional as F

class StochDepth(nn.Module):
    """Batchwise Dropout used in EfficientNet, optionally sans rescaling."""

    def __init__(self, drop_rate, scale_by_keep=False):
        super().__init__()
        self.drop_rate = drop_rate
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if not self.training:
            return x

        batch_size = x.shape[0]
        r = torch.rand([batch_size, 1, 1], dtype=x.dtype, device=x.device)
        keep_prob = 1.0 - self.drop_rate
        binary_tensor = torch.floor(keep_prob + r)
        if self.scale_by_keep:
            x = x / keep_prob
        return x * binary_tensor
    
class PosEmbed(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        self.pos_embed = nn.Parameter(
            torch.empty(input_size, dtype=torch.float32)
        )
        torch.nn.init.trunc_normal_(self.pos_embed,mean=0.0, std=0.02)

    def forward(self, x):
        return x + self.pos_embed.unsqueeze(0)
    
class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, stochdepth_rate):
        super().__init__()
        self.stochdepth_rate = stochdepth_rate
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, input_dim)
        if stochdepth_rate > 0.0:
            self.stochdepth = StochDepth(stochdepth_rate, scale_by_keep=True)
        else:
            self.stochdepth = None

    def forward(self, x):
        out = F.gelu(self.fc1(x))
        if self.stochdepth:
            out = self.stochdepth(out)
        out = self.fc2(out)
        return out
    
class SkipInitChannelwise(nn.Module):
    def __init__(self, channels, init_val=1e-6):
        super().__init__()
        self.channels = channels
        self.init_val = init_val
        self.skip = nn.Parameter(torch.ones(channels) * init_val)

    def forward(self, x):
        return x * self.skip
    
class ViTBlock(nn.Module):
    def __init__(self, input_dim, heads, key_dim, mlp_dim, layerscale_init, stochdepth_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim,eps=1e-3)
        self.attn = nn.MultiheadAttention(input_dim, heads, batch_first=True)
        self.skip1 = SkipInitChannelwise(key_dim, init_val=layerscale_init)
        self.stochdepth1 = StochDepth(stochdepth_rate, scale_by_keep=True) if stochdepth_rate > 0.0 else None
        self.norm2 = nn.LayerNorm(input_dim,eps=1e-3)
        self.mlp = MLPBlock(input_dim, mlp_dim, stochdepth_rate)
        self.skip2 = SkipInitChannelwise(key_dim, init_val=layerscale_init)
        self.stochdepth2 = StochDepth(stochdepth_rate, scale_by_keep=True) if stochdepth_rate > 0.0 else None

    def forward(self, x):
        out = self.norm1(x)
        out = self.attn(out, out, out)[0]
        out = self.skip1(out)
        if self.stochdepth1:
            out = self.stochdepth1(out)
        x = out + x

        out = self.norm2(x)
        out = self.mlp(out)
        out = self.skip2(out)
        if self.stochdepth2:
            out = self.stochdepth2(out)

        out = out + x
        return out
    
class ViT(nn.Module):
    def __init__(self, in_channels=3, img_size=320, out_classes=2000, definition_name="B16"):
        super().__init__()
        self.definitions = {
            "B16": {
                "num_blocks": 12,
                "patch_size": 16,
                "key_dim": 768,
                "mlp_dim": 3072,
                "heads": 12,
                "stochdepth_rate": 0.05,
            },
            # Other definitions removed for simplicity
        }

        definition = self.definitions[definition_name]
        self.blocks = nn.ModuleList()
        num_blocks = definition["num_blocks"]
        patch_size = definition["patch_size"]
        key_dim = definition["key_dim"]
        mlp_dim = definition["mlp_dim"]
        heads = definition["heads"]
        stochdepth_rate = definition["stochdepth_rate"]
        layerscale_init = 0.1  # Replacing CaiT_LayerScale_init(num_blocks)

        self.conv = nn.Conv2d(in_channels, key_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = PosEmbed(((img_size // patch_size) ** 2,key_dim))

        for i in range(num_blocks):
            self.blocks.append(
                ViTBlock(key_dim, heads, key_dim, mlp_dim, layerscale_init, stochdepth_rate)
            )

        self.norm = nn.LayerNorm(key_dim,eps=1e-3)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(key_dim, out_classes)
        self.act = nn.Sigmoid()

    def forward(self, x, return_pool: bool = False):
        x = (x - 127.5) / 127.5
        x = self.conv(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h*w).permute(0, 2, 1)  # (B, H*W, C)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2)).squeeze(-1)  # (B, C)
        
        if return_pool:
            return x # pfg feature
        
        x = self.fc(x)
        x = self.act(x)
        return x
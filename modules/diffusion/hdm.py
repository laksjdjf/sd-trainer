import torch
from .base import DiffusionModel

class HDMDiffusionModel(DiffusionModel):
    def bounding_box(self, h, w, pixel_aspect_ratio=1.0):
        # Adjusted dimensions
        w_adj = w
        h_adj = h * pixel_aspect_ratio

        # Adjusted aspect ratio
        ar_adj = w_adj / h_adj

        # Determine bounding box based on the adjusted aspect ratio
        y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
        if ar_adj > 1:
            y_min, y_max = -1 / ar_adj, 1 / ar_adj
        elif ar_adj < 1:
            x_min, x_max = -ar_adj, ar_adj

        return torch.tensor([y_min, y_max, x_min, x_max])

    def centers(self, start, stop, num, dtype=None, device=None):
        edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
        return (edges[:-1] + edges[1:]) / 2

    def make_grid(self, h_pos, w_pos):
        grid = torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1)
        return grid.flatten(0, 1)

    def make_axial_pos_no_cache(
        self, h, w, pixel_aspect_ratio=1.0, align_corners=False, dtype=None, device=None
    ):
        y_min, y_max, x_min, x_max = self.bounding_box(h, w, pixel_aspect_ratio)
        if align_corners:
            h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
            w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
        else:
            h_pos = self.centers(y_min, y_max, h, dtype=dtype, device=device)
            w_pos = self.centers(x_min, x_max, w, dtype=dtype, device=device)
        return self.make_grid(h_pos, w_pos)

    def forward(self, latents, timesteps, text_output, sample=False, size_condition=None, controlnet_hint=None):
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(latents.size(0))
        timesteps = timesteps.to(latents) / 1000
        latent_h, latent_w = latents.size(2), latents.size(3)
        
        aspect_ratio = (
            torch.tensor([latent_w / latent_h], device=latents.device)
            .log()
            .repeat(latents.size(0))
        ).to(latents.dtype)

        pos_map = self.make_axial_pos_no_cache(latent_h, latent_w, device=latents.device)[None].expand(latents.size(0), -1, -1).to(latents.dtype)

        model_output = self.unet(
            latents,
            timesteps,
            text_output.encoder_hidden_states,
            added_cond_kwargs={
                "addon_info": aspect_ratio,
                "tread_rate": 0.0,
            },
            pos_map=pos_map,
        ).sample

        return model_output

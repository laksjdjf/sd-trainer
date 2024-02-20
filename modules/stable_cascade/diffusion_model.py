import torch
import torch.nn as nn

class CascadeDiffusionModel(nn.Module):
    def __init__(
        self, 
        unet,
    ):
        super().__init__()
        self.unet = unet
    
    def forward(self, latents, timesteps, encoder_hidden_states, pooled_output, clip_image=None):
        ratios = (timesteps + 1) / 1000 # [0, 999] -> [0.001, 1.0]

        if ratios.dim() == 0:
            ratios = ratios.repeat(latents.shape[0])
        
        model_output = self.unet(
            x=latents,
            r=ratios,
            clip_text_pooled=pooled_output,
            clip_text=encoder_hidden_states,
            clip_img=clip_image,
        )

        return model_output
    
    def enable_gradient_checkpointing(self, enable:bool=True):
        raise NotImplementedError
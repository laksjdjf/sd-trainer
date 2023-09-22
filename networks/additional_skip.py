import torch
import torch.nn as nn
import os
from utils.functions import save_sd, load_sd

SDV1_CHANNELS = [320] * 4 + [640] * 3 + [1280] * 5
SDXL_CHANNELS = [320] * 4 + [640] * 3 + [1280] * 2
SDV1_DOWN = [1] * 3 + [2] * 3 + [4] * 3 + [8] * 3
SDXL_DOWN = [1] * 3 + [2] * 3 + [4] * 3

class AdditionalSkip(nn.Module):
    def __init__(self, has_mid=True, is_sdxl=False, resume=None) -> None:
        super().__init__()
        self.channels = SDV1_CHANNELS if not is_sdxl else SDXL_CHANNELS
        self.down = SDV1_DOWN if not is_sdxl else SDXL_DOWN
        self.has_mid = has_mid
        if self.has_mid:
            self.channels.append(1280)
            self.down.append(self.down[-1])
            
        self.params = nn.ParameterList([nn.Parameter(torch.zeros(1,channel,1,1)) for channel in self.channels])

        if resume is not None:
            self.load_state_dict(load_sd(resume))
    
    def res(self, batch_size, height, width):
        broadcast_param = [param.repeat(batch_size, 1, height//down, width//down) for param, down in zip(self.params, self.down)]
        down_block_additional_residuals = broadcast_param[:-1 if self.has_mid else None]
        mid_block_additional_residual = broadcast_param[-1] if self.has_mid else None
        
        return down_block_additional_residuals, mid_block_additional_residual
    
    def save_weights(self, path):
        if os.path.splitext(path)[1] == '':
            path += '.safetensors'

        save_sd(self.state_dict(), path)
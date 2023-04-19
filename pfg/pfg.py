import torch
import os

class PFGNetwork(torch.nn.Module):
    def __init__(self, input_size:int = 768, cross_attention_dim:int = 1024, num_tokens:int = 1) -> None:
        super().__init__()
        self.input_size = input_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        #input_dim -> token_dim * num_tokens
        self.pfg_linear = torch.nn.Linear(self.input_size, self.cross_attention_dim * self.num_tokens)
        
    def forward(self, x):
        x = self.pfg_linear(x)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        return x
    
    def save_weights(self, file, dtype=None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == '.safetensors':
            from safetensors.torch import save_file
            save_file(state_dict, file)
        else:
            torch.save(state_dict, file)

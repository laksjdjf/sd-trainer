import torch

class PFGNetwork(torch.nn.Module):
    def __init__(self, input_size:int = 768, cross_attention_dim:int = 1024, num_tokens:int = 1) -> None:
        super().__init__()
        self.input_size = input_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        #input_dim -> token_dim * num_tokens
        self.pfg_linear = torch.nn.Linear(self.input_size, cross_attention_dim * num_tokens)
        
    def forward(x):
        x = self.pfg_linear(x)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        return x

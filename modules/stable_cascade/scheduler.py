import torch
from modules.scheduler import BaseScheduler, substitution_t

class CosineScheduler(BaseScheduler):
    def __init__(self, v_prediction=False):
        self.v_prediction = v_prediction  # velocity予測かどうか
        self.make_alpha_beta()

    def make_alpha_beta(self, s=0.008, num_timesteps=1000, clamp_range=[0.001, 0.9999]):
        self.num_timesteps = num_timesteps

        # f_0, ... , f_T
        t_over_T = torch.linspace(1, self.num_timesteps+1, self.num_timesteps+1, dtype=torch.float32) / self.num_timesteps
        self.f0 = torch.cos(torch.tensor([s]) / (1 + torch.tensor([s])) * torch.pi * 0.5) ** 2
        self.f = torch.cos((t_over_T + s) / (1 + s) * torch.pi * 0.5).clamp(0, 1) **2

        # alpha_1, ... , alpha_T
        self.alphas_bar = (self.f / self.f0).clamp(clamp_range[0], clamp_range[1])
        self.betas_bar = 1 - self.alphas_bar

        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_betas_bar = torch.sqrt(self.betas_bar)

    def set_timesteps(self, num_inference_steps, device="cuda"):
        self.num_inference_steps = num_inference_steps

        timesteps = torch.linspace(0, self.num_timesteps, num_inference_steps+1, dtype=float).round()
        return timesteps.flip(0)[:-1].clone().long().to(device) # [999, ... , 0][:-1]
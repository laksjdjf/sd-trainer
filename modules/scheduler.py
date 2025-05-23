import torch
import math

def substitution_t(constants, timesteps, batch_size): 
    if timesteps.dim() == 0: # 全要素同じtの場合
        timesteps = timesteps.repeat(batch_size)
    device = timesteps.device
    constants = constants.to(device)[timesteps][:,None, None, None] # 4dims for latents
    return constants

class BaseScheduler:
    def __init__(self, v_prediction=False):
        self.v_prediction = v_prediction  # velocity予測かどうか
        self.make_alpha_beta()

    def make_alpha_beta(self, beta_start=0.00085, beta_end=0.012, num_timesteps=1000):
        self.num_timesteps = num_timesteps

        # beta_1, ... , beta_T
        self.betas = (torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32) ** 2)

        # alpha_1, ... , alpha_T
        self.alphas = 1 - self.betas

        # with bar
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.betas_bar = 1 - self.alphas_bar

        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_betas_bar = torch.sqrt(self.betas_bar)

    def set_timesteps(self, num_inference_steps, device="cuda"):
        self.num_inference_steps = num_inference_steps
        timesteps = torch.linspace(0, self.num_timesteps-1, num_inference_steps+1, dtype=float).round()
        return timesteps.flip(0)[:-1].clone().long().to(device) # [999, ... , 0][:-1]
    
    def sample_timesteps(self, batch_size, device="cuda", min_t=0, max_t=None):
        min_t = int(min_t * self.num_timesteps)
        max_t = int(max_t * self.num_timesteps) if max_t is not None else self.num_timesteps
        timesteps = torch.randint(min_t, max_t, (batch_size,), device=device)
        return timesteps
    
    def pred_original_sample(self, sample, model_output, t):
        sqrt_alphas_bar = substitution_t(self.sqrt_alphas_bar, t, sample.shape[0])
        sqrt_betas_bar = substitution_t(self.sqrt_betas_bar, t, sample.shape[0])
        
        if self.v_prediction:
            return sqrt_alphas_bar * sample - sqrt_betas_bar * model_output
        else: # noise_prediction
            return (sample - sqrt_betas_bar * model_output) / sqrt_alphas_bar
        
    def pred_noise(self, sample, model_output, t):
        sqrt_alphas_bar = substitution_t(self.sqrt_alphas_bar, t, model_output.shape[0])
        sqrt_betas_bar = substitution_t(self.sqrt_betas_bar, t, model_output.shape[0])
        
        if self.v_prediction:
            return sqrt_alphas_bar * model_output + sqrt_betas_bar * sample
        else: # noise_prediction
            return model_output

    # x0 -> xt    
    def add_noise(self, sample, noise, t):
        sqrt_alphas_bar = substitution_t(self.sqrt_alphas_bar, t, sample.shape[0])
        sqrt_betas_bar = substitution_t(self.sqrt_betas_bar, t, sample.shape[0])
        
        return sqrt_alphas_bar * sample + sqrt_betas_bar * noise
    
    # x_t -> x_prev_t
    def step(self, sample, model_output, t, prev_t):
        original_sample = self.pred_original_sample(sample, model_output, t)
        noise_pred = self.pred_noise(sample, model_output, t)

        return self.add_noise(original_sample, noise_pred, prev_t)
    
    def get_snr(self, t):
        alphas_bar = substitution_t(self.alphas_bar, t, t.shape[0])
        betas_bar = substitution_t(self.betas_bar, t, t.shape[0])
        return alphas_bar / betas_bar
    
    def get_target(self, sample, noise, t):
        if self.v_prediction:
            sqrt_alphas_bar = substitution_t(self.sqrt_alphas_bar, t, sample.shape[0])
            sqrt_betas_bar = substitution_t(self.sqrt_betas_bar, t, sample.shape[0])
            return sqrt_alphas_bar * noise - sqrt_betas_bar * sample
        else:
            return noise
        
def time_to_mult(t, sample):
    ret_t = t.float() / 1000
    ret_t = ret_t.to(sample)
    return ret_t.view(-1, 1, 1, 1)

class FlowScheduler:
    def __init__(self, shift=3.0):
        self.num_timesteps = 1000
        self.shift = shift
        return

    def set_timesteps(self, num_inference_steps, device="cuda"):
        self.num_inference_steps = num_inference_steps
        timesteps = torch.linspace(0, 1, num_inference_steps+1, dtype=float)
        timesteps = timesteps.flip(0)[:-1]
        timesteps = (timesteps * self.shift) / (1 + (self.shift - 1) * timesteps)
        timesteps *= 1000
        return timesteps.to(device)
    
    def sample_timesteps(self, batch_size, device="cuda", min_t=0, max_t=None):
        if min_t != 0 or max_t is not None:
            raise ValueError("min_t and max_t are not supported in FlowScheduler")
        logits_norm = torch.randn(batch_size, device=device)
        timesteps = logits_norm.sigmoid()
        timesteps = (timesteps * self.shift) / (1 + (self.shift - 1) * timesteps)
        return timesteps * 1000

    # x0 -> xt    
    def add_noise(self, sample, noise, t):
        mult = time_to_mult(t, sample)
        return sample * (1 - mult) + noise * mult
    
    # x_t -> x_prev_t
    def step(self, sample, model_output, t, prev_t):
        mult = time_to_mult(t, sample)
        prev_mult = time_to_mult(prev_t, sample)
        return sample + model_output * (prev_mult - mult)
    
    def pred_original_sample(self, sample, model_output, t):
        mult = time_to_mult(t, sample)
        return sample - model_output * mult
    
    def get_target(self, sample, noise, t):
        return noise - sample
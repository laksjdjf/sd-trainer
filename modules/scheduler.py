import torch

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
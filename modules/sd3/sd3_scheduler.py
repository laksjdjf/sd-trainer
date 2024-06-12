import torch

def time_to_mult(t, sample):
    ret_t = t.float() / 1000
    ret_t = ret_t.to(sample)
    return ret_t.view(-1, 1, 1, 1)

class SD3Scheduler:
    def __init__(self, v_prediction=False):
        self.v_prediction = v_prediction
        self.num_timesteps = 1000
        return

    def set_timesteps(self, num_inference_steps, device="cuda"):
        self.num_inference_steps = num_inference_steps
        timesteps = torch.linspace(0, self.num_timesteps-1, num_inference_steps+1, dtype=float).round()
        return timesteps.flip(0)[:-1].clone().long().to(device) # [999, ... , 0][:-1]
    
    def sample_timesteps(self, batch_size, device="cuda"):
        # must be changed 
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        return timesteps

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
        return sample - noise
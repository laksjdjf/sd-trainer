import torch
from omegaconf import OmegaConf
from utils.model import load_model, patch_mid_block_checkpointing
import sys
import subprocess
import importlib
from tqdm import tqdm

def get_attr_from_config(config_text: str):
    module = ".".join(config_text.split(".")[:-1])
    attr = config_text.split(".")[-1]
    return getattr(importlib.import_module(module), attr)

def get_gpu_memory_usage():
    cmd = ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    return int(result.stdout.decode('utf-8').strip())

def main(batch_size, config):
    text_model, vae, unet, scheduler = load_model("runwayml/stable-diffusion-v1-5")

    weight_dtype = get_attr_from_config(config.dtype)  
    train_dtype = weight_dtype if config.full_bf16 else torch.float32

    if config.xformers:
        unet.set_use_memory_efficient_attention_xformers(True)

    text_model.to("cuda" if not config.te_offload else "cpu", dtype=weight_dtype).eval()
    text_model.requires_grad_(False)
    vae.to("cuda" if not config.vae_offload else "cpu", dtype=weight_dtype).eval()
    vae.requires_grad_(False)

    if config.network.use:
        network_class = get_attr_from_config(config.network.module)
        network = network_class(text_model, unet, False, False, **config.network.args)
        network.to("cuda", dtype=train_dtype).train()
        network.requires_grad_(True)
        params = network.prepare_optimizer_params(1e-4, 1e-4)
        '''
        # freeze down?
        network.requires_grad_(False)
        params = []
        for name, param in network.named_parameters():
            if "lora_up" in name:
                param.requires_grad = True
                params.append(param)
        '''
        
        unet.to("cuda", dtype=weight_dtype).eval()
        unet.requires_grad_(False)
    else:
        unet.to("cuda", dtype=train_dtype).train()
        unet.requires_grad_(True)
        params = unet.parameters()

    if config.gradient_checkpointing:
        unet.train()
        unet.enable_gradient_checkpointing()
        patch_mid_block_checkpointing(unet.mid_block)
        
    optimizer_class = get_attr_from_config(config.optimizer)
    optimizer = optimizer_class(params,lr=1e-4)

    amp = weight_dtype != torch.float32 and not config.full_bf16

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    peak_memory = -10000
    progress_bar = tqdm(range(config.num_steps))
    for i in range(config.num_steps):
        latents = torch.randn(batch_size, 4, config.image_size//8, config.image_size//8).to("cuda", dtype=weight_dtype)
        if config.te_offload:
            encoder_hidden_states = torch.randn(batch_size, 77, 768).to("cuda", dtype=weight_dtype)
        else:
            encoder_hidden_states, _ = text_model.encode_text(["super saber fighting falcon"]*batch_size)
        timesteps = torch.randint(0, 1000, (batch_size,)).to("cuda", dtype=torch.long)
        noise = torch.randn_like(latents)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        with torch.autocast("cuda", enabled=amp, dtype=weight_dtype):
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        peak_memory = max(peak_memory, get_gpu_memory_usage()/1024)
        logs = {"loss": loss.item(), "peak_memory": f"{peak_memory: 2f}"}
        progress_bar.set_postfix(logs)
        progress_bar.update()

    return peak_memory
    
if __name__ == "__main__":
    config = OmegaConf.load(sys.argv[1])
    peak_memories = {}
    for batch_size in config.batch_sizes:
        torch.manual_seed(4545)
        torch.cuda.manual_seed(4545)
        peak_memories[batch_size] = f"{main(batch_size, config): .2f}"
        torch.cuda.empty_cache()
        
    for k, v in peak_memories.items():
        print(v)

from modules.trainer import BaseTrainer
from modules.flux.flux_diffusion_model import FluxDiffusionModel
from modules.flux.flux_text_model import FluxTextModel
from modules.sd3.sd3_scheduler import SD3Scheduler
from networks.manager import NetworkManager
from diffusers import AutoencoderKL, FluxTransformer2DModel
import logging

logger = logging.getLogger("トレーナーちゃん")

def load_model(path, sdxl=False, clip_skip=-2, revision=None, torch_dtype=None):
    text_model = FluxTextModel.from_pretrained(path, revision=revision, torch_dtype=torch_dtype)
    unet = FluxTransformer2DModel.from_pretrained(path, subfolder='transformer', revision=revision, torch_dtype=torch_dtype)
    vae = AutoencoderKL.from_pretrained(path, subfolder='vae', revision=revision, torch_dtype=torch_dtype)
    scheduler = None
            
    text_model.clip_skip = clip_skip
    return text_model, vae, unet, scheduler

class FluxTrainer(BaseTrainer):
    def __init__(self, config, diffusion:FluxTransformer2DModel, text_model:FluxTextModel, vae:AutoencoderKL, scheduler, network:NetworkManager):
        self.config = config
        self.diffusion = diffusion
        self.text_model = text_model
        self.vae = vae
        self.network = network
        self.diffusers_scheduler = scheduler # モデルのセーブ次にのみ利用
        self.scheduler = SD3Scheduler()
        self.sdxl = text_model.sdxl
        self.scaling_factor = 0.3611
        self.shift_factor = 0.1159
        self.input_channels = 16
    
    @classmethod
    def from_pretrained(cls, path, sdxl=False, clip_skip=None, config=None, network=None, revision=None, torch_dtype=None):
        if clip_skip is None:
            clip_skip = -2 if sdxl else -1
        text_model, vae, unet, scheduler = load_model(path, sdxl=False, clip_skip=-2, revision=revision, torch_dtype=torch_dtype)
        diffusion = FluxDiffusionModel(unet)
        return cls(config, diffusion, text_model, vae, scheduler, network)
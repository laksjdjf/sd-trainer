from modules.trainer import BaseTrainer
from modules.sd3.sd3_diffusion_model import SD3DiffusionModel
from modules.sd3.sd3_text_model import SD3TextModel
from modules.sd3.sd3_scheduler import SD3Scheduler
from networks.manager import NetworkManager
from diffusers import AutoencoderKL, SD3Transformer2DModel
import logging

logger = logging.getLogger("トレーナーちゃん")

def load_model(path, sdxl=False, clip_skip=-2, revision=None, torch_dtype=None):
    text_model = SD3TextModel.from_pretrained(path, revision=revision, torch_dtype=torch_dtype)
    unet = SD3Transformer2DModel.from_pretrained(path, subfolder='transformer', revision=revision, torch_dtype=torch_dtype)
    vae = AutoencoderKL.from_pretrained(path, subfolder='vae', revision=revision, torch_dtype=torch_dtype)
    scheduler = None
            
    text_model.clip_skip = clip_skip
    return text_model, vae, unet, scheduler

class SD3Trainer(BaseTrainer):
    def __init__(self, config, diffusion:SD3DiffusionModel, text_model:SD3TextModel, vae:AutoencoderKL, scheduler, network:NetworkManager):
        self.config = config
        self.diffusion = diffusion
        self.text_model = text_model
        self.vae = vae
        self.network = network
        self.diffusers_scheduler = scheduler # モデルのセーブ次にのみ利用
        self.scheduler = SD3Scheduler()
        self.sdxl = text_model.sdxl
        self.scaling_factor = 1.5305
        self.shift_factor = 0.0609
        self.input_channels = 16
    
    @classmethod
    def from_pretrained(cls, path, sdxl=False, clip_skip=None, config=None, network=None, revision=None, torch_dtype=None):
        if clip_skip is None:
            clip_skip = -2 if sdxl else -1
        text_model, vae, unet, scheduler = load_model(path, sdxl=False, clip_skip=-2, revision=revision, torch_dtype=torch_dtype)
        diffusion = SD3DiffusionModel(unet)
        return cls(config, diffusion, text_model, vae, scheduler, network)
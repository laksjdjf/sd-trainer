from modules.stable_cascade.text_model import CascadeTextModel
from modules.stable_cascade.modeling import EfficientNetEncoder, Previewer
from modules.utils import load_sd
from diffusers import DDPMScheduler
from diffusers.pipelines.stable_cascade.modeling_stable_cascade_common import StableCascadeUnet

def load_model(path, effnet_path, previwer_path):
    text_model = CascadeTextModel.from_pretrained(path)
    unet = StableCascadeUnet.from_pretrained(path, subfolder='prior')
    effnet = EfficientNetEncoder()
    effnet.load_state_dict(load_sd(effnet_path))
    previewer = Previewer()
    previewer.load_state_dict(load_sd(previwer_path))
    scheduler = DDPMScheduler.from_pretrained(path, subfolder='scheduler')

    return text_model, effnet, unet, scheduler, previewer
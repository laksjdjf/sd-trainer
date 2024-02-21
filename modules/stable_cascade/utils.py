from modules.stable_cascade.text_model import CascadeTextModel
from modules.stable_cascade.modeling import EfficientNetEncoder, Previewer, StageC
from modules.utils import load_sd
from diffusers import DDPMScheduler

def load_model(path):
    hf_path, unet_path, effnet_path, previwer_path = path.replace(" ", "").split(",")
    text_model = CascadeTextModel.from_pretrained(hf_path)
    unet = StageC()
    unet.load_state_dict(load_sd(unet_path))
    effnet = EfficientNetEncoder()
    effnet.load_state_dict(load_sd(effnet_path))
    previewer = Previewer()
    previewer.load_state_dict(load_sd(previwer_path))
    scheduler = DDPMScheduler.from_pretrained(hf_path, subfolder='scheduler')

    return text_model, effnet, unet, scheduler, previewer
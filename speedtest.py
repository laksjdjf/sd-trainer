from omegaconf import OmegaConf
import sys
import math
from accelerate.utils import set_seed
from modules.utils import get_attr_from_config, collate_fn
from modules.config import Config
from tqdm import tqdm
import logging
import subprocess
import time
import json
import pandas as pd
from itertools import product
import torch
import gc

logger = logging.getLogger("テストちゃん")

def get_gpu_memory_usage():
    cmd = ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    return int(result.stdout.decode('utf-8').strip())

def setattr_recursive(obj, key, value):
    if "." in key:
        key, rest = key.split(".", 1)
        setattr_recursive(getattr(obj, key), rest, value)
    else:
        setattr(obj, key, value)

def main(config):

    set_seed(config.main.seed)
    logger.info(f"シードは{config.main.seed}だよ！")
    
    logger.info(f"モデルを{config.main.model_path}からロードしちゃうよ！")
    trainer_cls = get_attr_from_config(config.trainer.module)
    trainer = trainer_cls.from_pretrained(config.main.model_path, config.main.sdxl, config.main.clip_skip, config.trainer)

    dataset_cls = get_attr_from_config(config.dataset.module)
    dataset = dataset_cls(trainer.text_model, **config.dataset.args)

    dataloder_cls = get_attr_from_config(config.dataloader.module)
    dataloader = dataloder_cls(dataset, collate_fn=collate_fn, **config.dataloader.args)

    trainer.prepare_modules_for_training()
    trainer.prepare_network(config.network)
    trainer.prepare_controlnet(config.controlnet)
    trainer.apply_module_settings()

    trainer.prepare_optimizer()

    steps_per_epoch = len(dataloader)
    total_steps = config.main.steps or steps_per_epoch * config.main.epochs
    total_epochs = config.main.epochs or math.floor(total_steps / steps_per_epoch)
    logger.info(f"トータルのステップ数は{total_steps}だよ！")

    trainer.prepare_lr_scheduler(total_steps)

    peek_memory = get_gpu_memory_usage()
    current_step = 0

    progress_bar = None
    for epoch in range(total_epochs):
        for batch in dataloader:
            if progress_bar is None:
                start_time = time.time()
                progress_bar = tqdm(total=total_steps, desc="Training")
            logs = trainer.step(batch)
            peek_memory = max(peek_memory, get_gpu_memory_usage())
            logs.update({"peek_memory": peek_memory})
            progress_bar.update(1)
            progress_bar.set_postfix(logs)
            current_step += 1

            if current_step == total_steps:
                logger.info(f"トレーニングが終わったよ！")
                end_time = time.time()
                seconds = end_time - start_time
                samples_per_second = total_steps*dataset.batch_size / seconds
                print(f"トータルの時間は{seconds:02}秒だよ！")
                print(f"VRAMのピークは{peek_memory}MBだよ!")
                print(f"1秒あたりのサンプル数は{samples_per_second}だよ！")
                del trainer.diffusion.unet, trainer.vae, trainer.text_model
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
                return seconds, total_steps, samples_per_second, peek_memory 

        logger.info(f"エポック{epoch+1}が終わったよ！")

if __name__ == "__main__":
    base_config = OmegaConf.load(sys.argv[1])
    base_config = OmegaConf.merge(OmegaConf.structured(Config), base_config)

    logging.basicConfig(level=logging.WARNING)
    print(OmegaConf.to_yaml(base_config))

    if len(sys.argv) == 3:
        with open(sys.argv[2], "r") as f:
            valiation = json.load(f)

        keys = list(valiation.keys())
        values = list(valiation.values())
        columns = [key.split(".")[-1] for key in keys]+["time", "steps", "samples/s", "vram", ]
        df = pd.DataFrame(columns=columns)

        for settings in product(*values):
            print({keys[i]: setting for i, setting in enumerate(settings)})
            for i, setting in enumerate(settings):
                setattr_recursive(base_config, keys[i], setting)
                
            try:
                seconds, steps, samples_par_second, memory  = main(base_config)
            except Exception as e:
                print(e)
                seconds, steps, samples_par_second, memory = 0, 0, 0, 0
            
            data = list(settings) + [seconds, steps, samples_par_second, memory]
            df.loc[len(df)] = data
            
        df.to_csv("speed_test.csv")

    else:
        main(base_config)
from omegaconf import OmegaConf
import sys
import math
from accelerate.utils import set_seed
from modules.utils import get_attr_from_config, collate_fn
from modules.config import Config
from tqdm import tqdm
import logging
import wandb
from rich.logging import RichHandler

logger = logging.getLogger("メインちゃん")

def main(config):

    set_seed(config.main.seed)
    logger.info(f"シードは{config.main.seed}だよ！")
    
    logger.info(f"モデルを{config.main.model_path}からロードしちゃうよ！")
    trainer_cls = get_attr_from_config(config.trainer.module)
    trainer = trainer_cls.from_pretrained(
        path = config.main.model_path, 
        model_type = config.main.model_type,
        revision = config.main.revision,
        variant = config.main.variant,
        nf4 = config.main.nf4,
        torch_dtype = get_attr_from_config(config.main.torch_dtype),
        clip_skip = config.main.clip_skip, 
        config = config.trainer
    )

    dataset_cls = get_attr_from_config(config.dataset.module)
    dataset = dataset_cls(trainer.text_model, **config.dataset.args)

    dataloder_cls = get_attr_from_config(config.dataloader.module)
    dataloader = dataloder_cls(dataset, collate_fn=collate_fn, **config.dataloader.args)

    trainer.prepare_modules_for_training()
    trainer.prepare_network(config.network)
    trainer.prepare_controlnet(config.controlnet)
    trainer.apply_module_settings()

    trainer.prepare_optimizer()

    if config.main.wandb is not None:
        wandb_run = wandb.init(project=config.main.wandb, name=config.main.output_path, dir="wandb")
    else:
        wandb_run = None

    steps_per_epoch = len(dataloader)
    total_steps = config.main.steps or steps_per_epoch * config.main.epochs
    total_epochs = config.main.epochs or math.floor(total_steps / steps_per_epoch)
    logger.info(f"トータルのステップ数は{total_steps}だよ！")

    trainer.prepare_lr_scheduler(total_steps)

    save_interval = config.main.save_steps or config.main.save_epochs * steps_per_epoch
    sample_interval = config.main.sample_steps or config.main.sample_epochs * steps_per_epoch
    logger.info(f"モデルを{save_interval}ステップごとにセーブするよ！")
    logger.info(f"サンプルは{sample_interval}ステップごとに生成するよ！")

    progress_bar = tqdm(total=total_steps, desc="Training")
    current_step = 0

    for epoch in range(total_epochs):
        for batch in dataloader:
            logs = trainer.step(batch)

            progress_bar.update(1)
            progress_bar.set_postfix(logs)
            if wandb_run is not None:
                wandb_run.log(logs, step=current_step)

            if current_step % save_interval == 0 or current_step == total_steps - 1:
                trainer.save_model(config.main.output_path)
            if current_step % sample_interval == 0 or current_step == total_steps - 1:
                images = trainer.sample_validation(batch)
                if wandb_run is not None:
                    images = [wandb.Image(image, caption=config.trainer.validation_args.prompt) for image in images]
                    wandb_run.log({'images': images}, step=current_step)
                else:
                    [image.save(f"image_logs/{current_step}_{i}.png") for i, image in enumerate(images)]

            current_step += 1

            if current_step == total_steps:
                logger.info(f"トレーニングが終わったよ！")
                if wandb_run is not None:
                    wandb_run.finish()
                return

        logger.info(f"エポック{epoch+1}が終わったよ！")

if __name__ == "__main__":
    config = OmegaConf.load(sys.argv[1])
    config = OmegaConf.merge(OmegaConf.structured(Config), config)
    logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(markup=True,rich_tracebacks=True)])
    print(OmegaConf.to_yaml(config))
    main(config)
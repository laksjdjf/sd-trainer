import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from accelerate.utils import set_seed
from modules.utils import get_attr_from_config, collate_fn
from modules.config import Config
import logging
from rich.logging import RichHandler

logger = logging.getLogger("メインちゃん")

# Configデータクラスをスキーマとして登録(conf/config.yamlのdefaults先頭で参照)
cs = ConfigStore.instance()
cs.store(name="base_schema", node=Config)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(config):
    logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]",
                        handlers=[RichHandler(markup=True, rich_tracebacks=True)], force=True)
    print(OmegaConf.to_yaml(config))

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
        taesd = config.main.taesd,
        torch_dtype = get_attr_from_config(config.main.torch_dtype),
        clip_skip = config.main.clip_skip,
        config = config.trainer
    )

    dataset_cls = get_attr_from_config(config.dataset.module)
    dataset = dataset_cls(trainer.text_model, **config.dataset.args)

    dataloder_cls = get_attr_from_config(config.dataloader.module)
    dataloader = dataloder_cls(dataset, collate_fn=collate_fn, **config.dataloader.args)

    logger.info(f"データセットのサイズは{len(dataset)}だよ！")

    trainer.prepare_modules_for_training()
    trainer.prepare_network(config.network)
    trainer.prepare_controlnet(config.controlnet)
    trainer.apply_module_settings()

    trainer.prepare_optimizer()

    trainer.fit(dataloader, config.main)

if __name__ == "__main__":
    main()

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from omegaconf import MISSING

@dataclass
class MainConfig:
    model_path: str = MISSING
    output_path: str = MISSING
    seed: Optional[int] = 4545
    sdxl: bool = MISSING
    clip_skip: Optional[bool] = None
    steps: Optional[int] = None
    epochs: Optional[int] = None
    save_steps: Optional[int] = None
    save_epochs: Optional[int] = 1
    sample_steps: Optional[int] = None
    sample_epochs: Optional[int] = 1
    log_level: str = "loggging.WARNING"
    wandb: Optional[str] = None

@dataclass
class OptimizerConfig:
    module: str = "torch.optim.AdamW"
    args: Optional[Any] = None

@dataclass
class TrainerConfig:
    module: str = "modules.trainer.BaseTrainer"
    train_unet: bool = MISSING
    train_text_encoder: bool = MISSING
    te_device: Optional[str] = None
    vae_device: Optional[str] = None
    train_dtype: str = MISSING
    weight_dtype: str = MISSING
    autocast_dtype: Optional[str] = None
    vae_dtype: Optional[str] = None
    lr: str = MISSING
    lr_scheduler: str = "constant"
    noise_offset: float = 0.0
    gradient_checkpointing: bool = False
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    merging_loras: Optional[List[str]] = None
    validation_num_samples: int = 4
    validation_seed: int = 4545
    validation_args: Dict[str, Any] = field(default_factory=dict)
    additional_conf: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DatasetConfig:
    module: str = MISSING
    args: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataLoaderArgs:
    num_workers: int = 0
    shuffle: bool = True

@dataclass
class DataLoaderConfig:
    module: str = MISSING
    args: DataLoaderArgs = field(default_factory=DataLoaderArgs)

@dataclass
class NetworkArgs:
    module: Optional[str] = None
    file_name: Optional[str] = None
    unet_key_filters: Optional[List[str]] = None
    module_args: Optional[Dict[str, Any]] = None
    conv_module_args: Optional[Dict[str, Any]] = None
    text_module_args: Optional[Dict[str, Any]] = None

@dataclass
class NetworkConfig:
    module: str = "networks.manager.NetworkManager"
    resume: Optional[str] = None
    train: bool = MISSING
    args: NetworkArgs = field(default_factory=NetworkArgs)

@dataclass
class ControlNetArgs:
    train: bool = MISSING
    resume: Optional[str] = None
    transformer_layers_per_block: Optional[List[int]] = None
    global_average_pooling: bool = False

@dataclass
class Config:
    main: MainConfig = field(default_factory=MainConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    network: Optional[NetworkConfig] = None
    controlnet: Optional[ControlNetArgs] = None
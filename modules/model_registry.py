# モデルタイプごとの分岐を一元管理するレジストリ。
#
# 各モデルタイプに対して「どうロードするか」「latentの統計量」「保存対応の有無」等を
# ModelSpec としてまとめ、modules/utils.py (load_model) や modules/trainer.py から
# 参照する。オプショナル依存 (hdm パッケージ) は該当ローダー関数の
# 内部で遅延importし、ここでは必須依存 (diffusers / modules.diffusion / modules.text /
# modules.scheduler) のみをトップレベルでimportする。
# modules.utils とは循環importになるため参照しない (get_attr_from_config は使わない)。
import math
import os
from dataclasses import dataclass, field
from typing import Callable

import torch
from diffusers import (
    AnimaTextConditioner,
    AuraFlowTransformer2DModel,
    AutoencoderKL,
    AutoencoderKLFlux2,
    AutoencoderKLHunyuanVideo,
    AutoencoderKLQwenImage,
    AutoencoderTiny,
    CosmosTransformer3DModel,
    DDPMScheduler,
    FlowMatchEulerDiscreteScheduler,
    Flux2Transformer2DModel,
    FluxTransformer2DModel,
    HunyuanVideoTransformer3DModel,
    Lumina2Transformer2DModel,
    SD3Transformer2DModel,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    ZImageTransformer2DModel,
)
from transformers import Qwen2Tokenizer, Qwen3Model, T5Tokenizer

from modules.diffusion import (
    AnimaDiffusionModel,
    AuraFlowDiffusionModel,
    DiffusionModel,
    Flux2KleinDiffusionModel,
    FluxDiffusionModel,
    HDMDiffusionModel,
    HunyuanVideoDiffusionModel,
    Lumina2DiffusionModel,
    SD3DiffusionModel,
    ZImageDiffusionModel,
)
from modules.text import (
    AnimaTextModel,
    AuraFlowTextModel,
    Flux2KleinTextModel,
    FluxTextModel,
    HDMTextModel,
    HunyuanVideoTextModel,
    Lumina2TextModel,
    SD1TextModel,
    SD3TextModel,
    SDXLTextModel,
    ZImageTextModel,
)
from modules.scheduler import BaseScheduler, FlowScheduler


@dataclass
class ModelSpec:
    model_type: str
    # (spec, path, clip_skip, revision, torch_dtype, variant, nf4_config, taesd)
    #   -> (text_model, vae, diffusion, diffusers_scheduler, scheduler)
    load: Callable
    # (vae) -> (scaling_factor, shift_factor)
    latent_stats: Callable
    input_channels: int
    default_clip_skip: int = -1
    supports_save_pretrained: bool = False

    # 標準ローダー (_load_standard) 用パラメータ。カスタムローダーを使うspec (sd1/sdxl/anima)
    # では未使用でよい。
    text_model_cls: type = None
    # 通常は diffusers の transformer クラスそのもの。hdm だけは外部パッケージへの
    # 遅延importが必要なため、zero-arg callable (呼び出すとクラスを返す) を入れる。
    unet_cls_name: object = None
    vae_cls: type = None
    diffusion_cls: type = None
    taesd_repo: str = None  # None なら taesd 未対応
    flow_shift: float = None
    # {"clip_skip", "quantization_config"} のサブセット。text_model.from_pretrained に
    # 追加で渡すkwarg名。
    text_extra_args: tuple = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# latent_stats 実装
# ---------------------------------------------------------------------------

def _const_latent_stats(scaling_factor, shift_factor):
    def _f(vae):
        return scaling_factor, shift_factor
    return _f


def _latent_stats_hdm(vae):
    scaling_factor = 1 / torch.tensor(vae.config.latents_std)[None, :, None, None]
    shift_factor = torch.tensor(vae.config.latents_mean)[None, :, None, None]
    return scaling_factor, shift_factor


def _latent_stats_flux2_klein(vae):
    scaling_factor = 1 / torch.sqrt(vae.bn.running_var.view(1, -1, 4, 1, 1).mean(dim=2) + 1e-4)
    shift_factor = vae.bn.running_mean.view(1, -1, 4, 1, 1).mean(dim=2)
    return scaling_factor, shift_factor


def _latent_stats_anima(vae):
    scaling_factor = 1 / torch.tensor(vae.config.latents_std)[None, :, None, None, None]
    shift_factor = torch.tensor(vae.config.latents_mean)[None, :, None, None, None]
    return scaling_factor, shift_factor


# ---------------------------------------------------------------------------
# ローダー: sd1 / sdxl (from_single_file 対応)
# ---------------------------------------------------------------------------

def _load_sd1(spec, path, clip_skip, revision, torch_dtype, variant, nf4_config, taesd):
    if os.path.isfile(path):
        pipe = StableDiffusionPipeline.from_single_file(path, scheduler_type="ddim")
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        unet = pipe.unet
        if taesd:
            vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch_dtype)
        else:
            vae = pipe.vae
        diffusers_scheduler = pipe.scheduler
        text_model = SD1TextModel(tokenizer, text_encoder, clip_skip=clip_skip)
        del pipe
    else:
        text_model = SD1TextModel.from_pretrained(path, clip_skip=clip_skip, revision=revision, torch_dtype=torch_dtype, variant=variant)
        unet = UNet2DConditionModel.from_pretrained(path, subfolder='unet', revision=revision, torch_dtype=torch_dtype, variant=variant, quantization_config=nf4_config)
        if taesd:
            vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch_dtype)
        else:
            vae = AutoencoderKL.from_pretrained(path, subfolder='vae', revision=revision, torch_dtype=torch_dtype, variant=variant)
        diffusers_scheduler = DDPMScheduler.from_pretrained(path, subfolder='scheduler', revision=revision)
    scheduler = BaseScheduler(diffusers_scheduler.config.prediction_type == "v_prediction")
    diffusion = DiffusionModel(unet)
    return text_model, vae, diffusion, diffusers_scheduler, scheduler


def _load_sdxl(spec, path, clip_skip, revision, torch_dtype, variant, nf4_config, taesd):
    if os.path.isfile(path):
        pipe = StableDiffusionXLPipeline.from_single_file(path, scheduler_type="ddim")
        tokenizer = pipe.tokenizer
        tokenizer_2 = pipe.tokenizer_2
        text_encoder = pipe.text_encoder
        text_encoder_2 = pipe.text_encoder_2
        unet = pipe.unet
        if taesd:
            vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch_dtype)
        else:
            vae = pipe.vae
        diffusers_scheduler = pipe.scheduler
        text_model = SDXLTextModel(tokenizer, tokenizer_2, text_encoder, text_encoder_2, clip_skip=clip_skip)
        del pipe
    else:
        text_model = SDXLTextModel.from_pretrained(path, clip_skip=clip_skip, revision=revision, torch_dtype=torch_dtype, variant=variant)
        unet = UNet2DConditionModel.from_pretrained(path, subfolder='unet', revision=revision, torch_dtype=torch_dtype, variant=variant, quantization_config=nf4_config)
        if taesd:
            vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch_dtype)
        else:
            vae = AutoencoderKL.from_pretrained(path, subfolder='vae', revision=revision, torch_dtype=torch_dtype, variant=variant)
        diffusers_scheduler = DDPMScheduler.from_pretrained(path, subfolder='scheduler', revision=revision)
    diffusion = DiffusionModel(unet, sdxl=True)
    scheduler = BaseScheduler(diffusers_scheduler.config.prediction_type == "v_prediction")
    return text_model, vae, diffusion, diffusers_scheduler, scheduler


# ---------------------------------------------------------------------------
# ローダー: sd3 / flux / lumina2 / auraflow / hunyuan_video / hdm / zimage / flux2_klein
# ---------------------------------------------------------------------------

def _import_hdm_transformer():
    try:
        from hdm import XUDiTConditionModel
    except ImportError as e:
        raise ImportError(
            "HDM support requires the hdm package. Install it in the Python environment used for training."
        ) from e
    return XUDiTConditionModel


def _load_standard(spec, path, clip_skip, revision, torch_dtype, variant, nf4_config, taesd):
    # unet_cls_name の解決 (hdm のような遅延importが必要なケースを含む) は、現行実装
    # (utils.py の hdm 分岐) に合わせてネットワークアクセスより前に行い、パッケージ
    # 未インストール時に明確な ImportError を優先して出す。
    unet_cls = spec.unet_cls_name
    if not isinstance(unet_cls, type):
        unet_cls = unet_cls()

    if os.path.isfile(path):
        raise NotImplementedError(f"from_single_file is not implemented for {spec.model_type}")

    text_kwargs = {}
    if "clip_skip" in spec.text_extra_args:
        text_kwargs["clip_skip"] = clip_skip
    if "quantization_config" in spec.text_extra_args:
        text_kwargs["quantization_config"] = nf4_config
    text_model = spec.text_model_cls.from_pretrained(
        path, revision=revision, torch_dtype=torch_dtype, variant=variant, **text_kwargs
    )

    unet = unet_cls.from_pretrained(
        path, subfolder='transformer', revision=revision, torch_dtype=torch_dtype, variant=variant,
        quantization_config=nf4_config,
    )

    if taesd and spec.taesd_repo:
        vae = AutoencoderTiny.from_pretrained(spec.taesd_repo, torch_dtype=torch_dtype)
    else:
        if taesd and spec.taesd_repo is None and spec.model_type == "auraflow":
            Warning("taesd is not implemented for AuraFlow")
        vae = spec.vae_cls.from_pretrained(path, subfolder='vae', revision=revision, torch_dtype=torch_dtype, variant=variant)

    diffusers_scheduler = None
    scheduler = FlowScheduler(shift=spec.flow_shift)
    diffusion = spec.diffusion_cls(unet)
    return text_model, vae, diffusion, diffusers_scheduler, scheduler


# ---------------------------------------------------------------------------
# ローダー: anima (公式Diffusers形式の各コンポーネントを直接ロード)
# ---------------------------------------------------------------------------

def _load_anima(spec, path, clip_skip, revision, torch_dtype, variant, nf4_config, taesd):
    if os.path.isfile(path):
        raise ValueError(
            "Official Diffusers Anima support requires a Diffusers-format model directory or Hub repository; "
            "single-file checkpoints are not supported."
        )

    model_kwargs = {
        "revision": revision,
        "torch_dtype": torch_dtype,
        "variant": variant,
    }
    tokenizer_kwargs = {"revision": revision}

    tokenizer = Qwen2Tokenizer.from_pretrained(path, subfolder="tokenizer", **tokenizer_kwargs)
    t5_tokenizer = T5Tokenizer.from_pretrained(path, subfolder="t5_tokenizer", **tokenizer_kwargs)
    text_encoder = Qwen3Model.from_pretrained(path, subfolder="text_encoder", **model_kwargs)
    text_conditioner = AnimaTextConditioner.from_pretrained(path, subfolder="text_conditioner", **model_kwargs)
    unet = CosmosTransformer3DModel.from_pretrained(path, subfolder="transformer", **model_kwargs)
    vae = AutoencoderKLQwenImage.from_pretrained(path, subfolder="vae", **model_kwargs)
    diffusers_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        path,
        subfolder="scheduler",
        revision=revision,
    )

    text_model = AnimaTextModel(
        tokenizer=tokenizer,
        t5_tokenizer=t5_tokenizer,
        text_encoder=text_encoder,
        text_conditioner=text_conditioner,
    )
    scheduler = FlowScheduler(shift=getattr(diffusers_scheduler.config, "shift", 3.0))
    diffusion = AnimaDiffusionModel(unet)
    return text_model, vae, diffusion, diffusers_scheduler, scheduler


# ---------------------------------------------------------------------------
# レジストリ本体
# ---------------------------------------------------------------------------

MODEL_SPECS: dict = {
    "sd1": ModelSpec(
        model_type="sd1",
        load=_load_sd1,
        latent_stats=_const_latent_stats(0.18215, 0),
        input_channels=4,
        default_clip_skip=-1,
        supports_save_pretrained=True,
    ),
    "sdxl": ModelSpec(
        model_type="sdxl",
        load=_load_sdxl,
        latent_stats=_const_latent_stats(0.13025, 0),
        input_channels=4,
        default_clip_skip=-2,
        supports_save_pretrained=True,
    ),
    "sd3": ModelSpec(
        model_type="sd3",
        load=_load_standard,
        latent_stats=_const_latent_stats(1.5305, 0.0609),
        input_channels=16,
        default_clip_skip=-1,
        text_model_cls=SD3TextModel,
        unet_cls_name=SD3Transformer2DModel,
        vae_cls=AutoencoderKL,
        diffusion_cls=SD3DiffusionModel,
        taesd_repo="madebyollin/taesd3",
        flow_shift=3.0,
        text_extra_args=("clip_skip",),
    ),
    "flux": ModelSpec(
        model_type="flux",
        load=_load_standard,
        latent_stats=_const_latent_stats(0.3611, 0.1159),
        input_channels=16,
        default_clip_skip=-1,
        text_model_cls=FluxTextModel,
        unet_cls_name=FluxTransformer2DModel,
        vae_cls=AutoencoderKL,
        diffusion_cls=FluxDiffusionModel,
        taesd_repo="madebyollin/taef1",
        flow_shift=math.exp(1.15),
        text_extra_args=(),
    ),
    "lumina2": ModelSpec(
        model_type="lumina2",
        load=_load_standard,
        latent_stats=_const_latent_stats(0.3611, 0.1159),
        input_channels=16,
        default_clip_skip=-2,
        text_model_cls=Lumina2TextModel,
        unet_cls_name=Lumina2Transformer2DModel,
        vae_cls=AutoencoderKL,
        diffusion_cls=Lumina2DiffusionModel,
        taesd_repo="madebyollin/taef1",
        flow_shift=6.0,
        text_extra_args=(),
    ),
    "auraflow": ModelSpec(
        model_type="auraflow",
        load=_load_standard,
        latent_stats=_const_latent_stats(0.13025, 0),
        input_channels=4,
        default_clip_skip=-1,
        text_model_cls=AuraFlowTextModel,
        unet_cls_name=AuraFlowTransformer2DModel,
        vae_cls=AutoencoderKL,
        diffusion_cls=AuraFlowDiffusionModel,
        taesd_repo=None,  # taesd指定時はWarningのみで通常vaeを使う
        flow_shift=1.73,
        text_extra_args=(),
    ),
    "hunyuan_video": ModelSpec(
        model_type="hunyuan_video",
        load=_load_standard,
        latent_stats=_const_latent_stats(0.476986, 0),
        input_channels=16,
        default_clip_skip=-3,
        text_model_cls=HunyuanVideoTextModel,
        unet_cls_name=HunyuanVideoTransformer3DModel,
        vae_cls=AutoencoderKLHunyuanVideo,
        diffusion_cls=HunyuanVideoDiffusionModel,
        taesd_repo=None,
        flow_shift=math.exp(1.15),
        text_extra_args=("quantization_config",),
    ),
    "hdm": ModelSpec(
        model_type="hdm",
        load=_load_standard,
        latent_stats=_latent_stats_hdm,
        input_channels=4,
        default_clip_skip=-1,
        text_model_cls=HDMTextModel,
        unet_cls_name=_import_hdm_transformer,
        vae_cls=AutoencoderKL,
        diffusion_cls=HDMDiffusionModel,
        taesd_repo=None,
        flow_shift=1.0,
        text_extra_args=(),
    ),
    "zimage": ModelSpec(
        model_type="zimage",
        load=_load_standard,
        latent_stats=_const_latent_stats(0.3611, 0.1159),
        input_channels=16,
        default_clip_skip=-2,
        text_model_cls=ZImageTextModel,
        unet_cls_name=ZImageTransformer2DModel,
        vae_cls=AutoencoderKL,
        diffusion_cls=ZImageDiffusionModel,
        taesd_repo=None,
        flow_shift=math.exp(1.15),
        text_extra_args=(),
    ),
    "flux2_klein": ModelSpec(
        model_type="flux2_klein",
        load=_load_standard,
        latent_stats=_latent_stats_flux2_klein,
        input_channels=32,
        default_clip_skip=-1,
        text_model_cls=Flux2KleinTextModel,
        unet_cls_name=Flux2Transformer2DModel,
        vae_cls=AutoencoderKLFlux2,
        diffusion_cls=Flux2KleinDiffusionModel,
        taesd_repo=None,
        flow_shift=math.exp(1.15),
        text_extra_args=(),
    ),
    "anima": ModelSpec(
        model_type="anima",
        load=_load_anima,
        latent_stats=_latent_stats_anima,
        input_channels=16,
        default_clip_skip=-1,
        supports_save_pretrained=False,
    ),
}


def get_model_spec(model_type: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_type]
    except KeyError:
        available = ", ".join(sorted(MODEL_SPECS.keys()))
        raise ValueError(f"Unknown model_type: {model_type!r}. Available types: {available}")

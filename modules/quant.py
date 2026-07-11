"""ベースモデルのLinear重みをcomfy_kitchenのINT8+ConvRot量子化テンソルに差し替える。

LoRA学習(QLoRA的な使い方)向け: ベース重みは凍結のままINT8化してVRAMを削減し、
forward(勾配チェックポイント再計算含む)をINT8テンサーコアで高速化する。
勾配はLoRA側のみに流れる前提のため、train_unet=Trueとは併用不可。

層選択はComfy-Org公式quant_int8_convrot.pyと同じdenylist方式。
"""
import re
import logging

import torch

try:
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_weight, _rotate_activation
except ImportError:
    from comfy_kitchen.tensor.int8 import _build_hadamard, _rotate_weight, _rotate_activation

logger = logging.getLogger("量子化ちゃん")

VALID_GS = (256, 64, 16)  # ConvRotのHadamardサイズ(4の冪)。大きい順に優先

# 公式quant_int8_convrot.pyと同じ除外パターン(embedder/timestep/head/adapter等)
EXCLUDE_SEG = re.compile(
    r"scale_shift|rope|rotary|rel_pos|pos_?embed|embedder|"
    r"gate_logits|router|routing|logit|temperature|"
    r"(?:^|_)time|temb|t_emb|guidance|register|refiner_blocks|adapter|"
    r"(?:^|_)(?:final|head|proj_out|out_layer)(?:_|$)")


def _best_gs(k):
    return next((g for g in VALID_GS if k % g == 0), None)


def _eligible(name, module, min_gemm):
    if not isinstance(module, torch.nn.Linear):
        return None
    n, k = module.weight.shape
    if n < 8 or min(n, k) < min_gemm:
        return None
    gs = _best_gs(k)
    if gs is None:
        return None
    segs = name.split(".")
    # インデックス付きブロック内の層のみ対象(公式スクリプトと同じ判定)
    if not any(segs[i].isdigit() for i in range(len(segs) - 1)):
        return None
    if any(EXCLUDE_SEG.search(s) for s in segs):
        return None
    return gs


@torch.no_grad()
def _build_bwd_copy(w, gs):
    """backward(grad_x = g @ W_rot)用の転置int8コピーを作る。

    forward側はW_rotを出力ch(n)単位スケールで持つが、backwardの縮約はn軸なので
    そのままではスケールを総和の外に出せない。転置[K,N]を入力ch(k)単位スケールで
    量子化し直すと、backwardが「per-row動的量子化の活性 × per-rowスケール重み」という
    forwardのint8_linearと同型のGEMMになり、既存カーネルをそのまま使える。
    """
    wf = w.detach().float()
    h = _build_hadamard(gs, device=wf.device, dtype=torch.float32)
    w_rot_t = _rotate_weight(wf, h, gs).t().contiguous()  # [K, N]
    scale = (w_rot_t.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
    q = (w_rot_t / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale.reshape(-1).float()


class _Int8ConvRotBwdLinearFn(torch.autograd.Function):
    """forward=既存のint8+ConvRotディスパッチ、backward=転置int8コピーでgrad_xもint8化。

    重み凍結(LoRA学習)専用: grad_weight/grad_biasは計算しない。
    grad_x_rot = int8_linear(g, W_rot_qT) を回転基底で計算し、Hadamardが対称
    (H^T == H)なことを利用して同じ回転をもう一度かけて元の基底に戻す。
    """

    @staticmethod
    def forward(ctx, x, qweight, w_bwd, w_bwd_scale, bias, gs):
        ctx.gs = gs
        ctx.save_for_backward(w_bwd, w_bwd_scale)
        return torch.nn.functional.linear(x, qweight, bias)

    @staticmethod
    def backward(ctx, g):
        grad_x = None
        if ctx.needs_input_grad[0]:
            from comfy_kitchen.registry import registry
            w_bwd, w_bwd_scale = ctx.saved_tensors
            gs = ctx.gs
            g2d = g.reshape(-1, g.shape[-1]).contiguous()
            impl = registry.get_implementation(
                "int8_linear",
                kwargs={"x": g2d, "weight": w_bwd, "weight_scale": w_bwd_scale, "bias": None},
            )
            grad_rot = impl(g2d, w_bwd, w_bwd_scale, bias=None, out_dtype=g.dtype,
                            convrot=False, convrot_groupsize=gs)
            h = _build_hadamard(gs, device=grad_rot.device, dtype=grad_rot.dtype)
            grad_x = _rotate_activation(grad_rot, h, gs)
            grad_x = grad_x.reshape(*g.shape[:-1], w_bwd.shape[0])
        return grad_x, None, None, None, None, None


def _install_int8_backward(module, gs):
    q, scale = _build_bwd_copy(module.weight, gs)
    module.register_buffer("weight_bwd", q, persistent=False)
    module.register_buffer("weight_bwd_scale", scale, persistent=False)

    def forward(x):
        return _Int8ConvRotBwdLinearFn.apply(
            x, module.weight, module.weight_bwd, module.weight_bwd_scale, module.bias, gs)

    module.forward = forward


@torch.no_grad()
def quantize_unet_int8_convrot(unet, min_gemm=256, exclude=None, int8_backward=False):
    """unet内の対象LinearのweightをINT8+ConvRotのQuantizedTensorへ置換する。

    ベース重みが凍結(requires_grad=False)であることが前提。unetは事前に
    デバイス/dtype確定済み(.to()適用後)であること — QuantizedTensor化した後の
    モデル全体.to(dtype)は想定外の経路を踏む可能性がある。
    """
    from comfy_kitchen.tensor import QuantizedTensor

    exc = re.compile(exclude) if exclude else None
    n_replaced = 0
    bytes_before = 0
    bytes_after = 0
    for name, module in unet.named_modules():
        gs = _eligible(name, module, min_gemm)
        if gs is None or (exc and exc.search(name)):
            continue
        w = module.weight
        if w.requires_grad:
            raise RuntimeError(
                f"{name}: 量子化対象の重みがrequires_grad=Trueです。"
                "quantize_unetはベース凍結(LoRA学習)専用です。train_unet=Falseにしてください。")
        qw = QuantizedTensor.from_float(
            w.detach(), "TensorWiseINT8Layout",
            per_channel=True, convrot=True, convrot_groupsize=gs)
        bytes_before += w.numel() * w.element_size()
        bytes_after += qw._qdata.numel() + qw._params.scale.numel() * 4
        if int8_backward:
            _install_int8_backward(module, gs)  # 元のbf16重みから転置コピーを作るため置換前に呼ぶ
            bytes_after += module.weight_bwd.numel() + module.weight_bwd_scale.numel() * 4
        module.weight = torch.nn.Parameter(qw, requires_grad=False)
        n_replaced += 1

    logger.info(
        f"Linear {n_replaced}層をINT8+ConvRotに量子化したよ！"
        f"{'(int8 backwardつき)' if int8_backward else ''}"
        f" 重みメモリ {bytes_before / 1e9:.2f}GB -> {bytes_after / 1e9:.2f}GB")
    if n_replaced == 0:
        logger.warning("量子化対象の層が見つからなかったよ。min_gemm/モデル構造を確認してね。")
    return n_replaced

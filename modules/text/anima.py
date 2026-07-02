import torch
import torch.nn as nn

from .base import BaseTextModel, BaseTextOutput


class AnimaTextModel(BaseTextModel):
    def __init__(
        self,
        prompt_tokenizer,
        text_encoder: nn.Module,
        max_sequence_length: int = 512,
    ):
        super().__init__()
        self.tokenizers = [prompt_tokenizer]
        self.text_encoders = nn.ModuleList([text_encoder])
        self.max_sequence_length = max_sequence_length
        self.clip_skip = None

    def forward(self, prompts):
        # diffusers_animaはオプショナル依存のため、import失敗でパッケージ全体を壊さないよう使用時に読み込む
        from diffusers_anima.pipelines.anima.text_encoding import prepare_condition_inputs

        prompts = [prompts] if isinstance(prompts, str) else prompts
        device = self.text_encoders[0].device
        dtype = self.dtype

        qwen_hidden, t5_ids, t5_weights = prepare_condition_inputs(
            self.tokenizers[0],
            self.text_encoders[0],
            prompts,
            execution_device=str(device),
            model_dtype=dtype,
        )
        return BaseTextOutput(qwen_hidden, t5_ids, t5_weights)

    def requires_grad_(self, requires_grad: bool = True):
        self.text_encoders.requires_grad_(requires_grad)
        return self

    @property
    def dtype(self):
        return next(self.text_encoders[0].parameters()).dtype

import torch
import torch.nn as nn

from .base import BaseTextModel, BaseTextOutput


class AnimaTextModel(BaseTextModel):
    def __init__(
        self,
        tokenizer,
        t5_tokenizer,
        text_encoder: nn.Module,
        text_conditioner: nn.Module,
        max_sequence_length: int = 512,
    ):
        super().__init__()
        self.tokenizers = [tokenizer]
        self.t5_tokenizer = t5_tokenizer
        self.text_encoders = nn.ModuleList([text_encoder])
        self.text_conditioner = text_conditioner
        self.text_conditioner.requires_grad_(False)
        self.max_sequence_length = max_sequence_length
        self.clip_skip = None

    def forward(self, prompts):
        prompts = [prompts] if isinstance(prompts, str) else prompts
        device = self.text_encoders[0].device
        dtype = self.dtype

        qwen_inputs = self.tokenizers[0](
            prompts,
            padding="longest",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        qwen_input_ids = qwen_inputs.input_ids.to(device)
        qwen_attention_mask = qwen_inputs.attention_mask.to(device)
        if qwen_input_ids.shape[-1] == 0:
            qwen_input_ids = qwen_input_ids.new_zeros((qwen_input_ids.shape[0], 1))
            qwen_attention_mask = qwen_attention_mask.new_zeros((qwen_attention_mask.shape[0], 1))

        t5_inputs = self.t5_tokenizer(
            prompts,
            padding="longest",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        # AnimaのQwen encoderとtext conditionerは学習対象外。conditionerの出力を
        # cloneして、inference tensorをdenoiserのautogradへ安全に渡す。
        with torch.inference_mode():
            qwen_hidden = self.text_encoders[0](
                input_ids=qwen_input_ids,
                attention_mask=qwen_attention_mask,
                output_hidden_states=False,
            ).last_hidden_state
            qwen_hidden = qwen_hidden.to(device=device, dtype=dtype)
            qwen_hidden = qwen_hidden * qwen_attention_mask.to(qwen_hidden).unsqueeze(-1)

            prompt_embeds = self.text_conditioner(
                source_hidden_states=qwen_hidden.to(dtype=self.text_conditioner.dtype),
                target_input_ids=t5_inputs.input_ids.to(device),
                target_attention_mask=t5_inputs.attention_mask.to(device),
                source_attention_mask=qwen_attention_mask,
            )

        return BaseTextOutput(prompt_embeds.to(device=device, dtype=dtype).clone())

    def requires_grad_(self, requires_grad: bool = True):
        self.text_encoders.requires_grad_(requires_grad)
        self.text_conditioner.requires_grad_(False)
        return self

    @property
    def dtype(self):
        return next(self.text_encoders[0].parameters()).dtype

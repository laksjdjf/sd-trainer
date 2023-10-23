import torch
from utils.functions import default

class TextEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, prompts, text_model, device="cuda", batch_size=1, clip_skip=-1):
        self.text_embedding_list = []
        self.batch_size = batch_size
        self.device = device
        for prompt in prompts:
            target = prompt["target"]
            positive = default(prompt, "positive", target)
            negative = default(prompt, "negative", "")
            neutral = default(prompt, "neutral", "")
            no_neutral = negative == neutral
            guidance_scale = default(prompt, "guidance_scale", 1)

            texts = [target, positive, negative, neutral] # 冗長な""の計算とかなくしたい気も知るけど別にいいか 

            tokens, tokens_2, empty_text = text_model.tokenize(texts)
            encoder_hidden_states, pooled_output = text_model(tokens, tokens_2, empty_text)
            if pooled_output is not None:
                target_proj, positive_proj, negative_proj, neutral_proj = pooled_output.chunk(4)
            else:
                target_proj, positive_proj, negative_proj, neutral_proj = None, None, None, None
            
            target, positive, negative, neutral = encoder_hidden_states.chunk(4)
            text_dict = {
                "target": (target, target_proj),
                "positive": (positive, positive_proj), 
                "negative": (negative, negative_proj), 
                "neutral": (neutral, neutral_proj) if not no_neutral else None,
                "guidance_scale": guidance_scale
            }
            self.text_embedding_list.append(text_dict)

    def __len__(self):
        return len(self.text_embedding_list)

    def __getitem__(self, idx):
        batch = {}
        for key in ["target", "positive", "negative", "neutral"]:
            if self.text_embedding_list[idx][key] is not None:
                text_emb = self.text_embedding_list[idx][key][0].repeat(self.batch_size, 1, 1).to(self.device)
                if self.text_embedding_list[idx][key][1] is not None:
                    projection = self.text_embedding_list[idx][key][1].repeat(self.batch_size, 1).to(self.device)
                else:
                    projection = None
                batch[key] = (text_emb, projection)
            else:
                batch[key] = (None, None)
        for key in ["guidance_scale"]:
            batch[key] = self.text_embedding_list[idx][key]
        return batch

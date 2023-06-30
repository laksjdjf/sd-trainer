import torch

# 辞書にkeyがあればその値を、なければdefault_valueを返す
def default(dic, key, default_value):
    if key in dic and dic[key] is not None:
        return dic[key]
    else:
        return default_value

class TextEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, prompts, tokenizer, text_encoder, device, batch_size=1):
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

            tokens = tokenizer(texts, max_length=tokenizer.model_max_length, padding="max_length",
                        truncation=True, return_tensors='pt').input_ids.to(self.device)
            encoder_hidden_states = text_encoder(tokens, output_hidden_states=True).last_hidden_state.detach().float().cpu()
            target, positive, negative, neutral = encoder_hidden_states.chunk(4)
            text_dict = {
                "target": target,
                "positive": positive, 
                "negative": negative, 
                "neutral": neutral if not no_neutral else None,
                "guidance_scale": guidance_scale
            }
            self.text_embedding_list.append(text_dict)

    def __len__(self):
        return len(self.text_embedding_list)

    def __getitem__(self, idx):
        batch = {}
        for key in ["target", "positive", "negative", "neutral"]:
            if self.text_embedding_list[idx][key] is not None:
                batch[key] = self.text_embedding_list[idx][key].repeat(self.batch_size, 1, 1).to(self.device)
            else:
                batch[key] = None
        for key in ["guidance_scale"]:
            batch[key] = self.text_embedding_list[idx][key]
        return batch
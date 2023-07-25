import torch

# 辞書にkeyがあればその値を、なければdefault_valueを返す
def default(dic, key, default_value):
    if key in dic and dic[key] is not None:
        return dic[key]
    else:
        return default_value

class TextEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, prompts, tokenizer, text_encoder, tokenizer_2=None, text_encoder_2=None, device="cuda", batch_size=1, clip_skip=-1):
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
            encoder_hidden_states = text_encoder(tokens, output_hidden_states=True)
            if tokenizer_2 is None:
                encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.hidden_states[clip_skip].detach().float().cpu()
            if tokenizer_2 is not None:
                tokens_2 = tokenizer_2(texts, max_length=tokenizer_2.model_max_length, padding="max_length",
                            truncation=True, return_tensors='pt').input_ids.to(self.device)
                encoder_output_2 = text_encoder_2(tokens_2, output_hidden_states=True)

                encoder_hidden_states_2 = encoder_output_2.hidden_states[clip_skip].detach().float().cpu()
                encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_2], dim=-1)
                
                projection = encoder_output_2[0].detach().float().cpu()
                for i, text in enumerate(texts):
                    if text == "":
                        projection[i] *= 0 # zero vector for empty text
                target_proj, positive_proj, negative_proj, neutral_proj = projection.chunk(4)
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
                batch[key] = None
        for key in ["guidance_scale"]:
            batch[key] = self.text_embedding_list[idx][key]
        return batch
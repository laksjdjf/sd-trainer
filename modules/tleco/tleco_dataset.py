from torch.utils.data import Dataset
import random

class TLECODataset(Dataset):
    def __init__(
        self, 
        text_model, 
        path, # 置き換え対象のテキストが書かれたファイルのパス
        batch_size=1,
        pad_tokens=None, # パディングトークンの文字列リスト
        selected_tags=None, # 追加するタグのリストが書かれたファイルのパス
        drop_rate=0.0, # 置き換え対象をドロップする確率（正則化）
        shuffle=True,
        prefix=None, # 先頭文字列
        repeat=1, # データセットの反復回数
        max_add_length=16, # 追加するタグの最大数
        position_rate=0.5, # タグの位置を先頭にする確率(1-rで末尾）
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_rate = drop_rate
        self.prefix = prefix
        self.max_add_length = max_add_length
        self.position_rate = position_rate

        with open(path, 'r') as f:
            self.texts = f.read().splitlines()
        self.text_pairs = [[tag.strip() for tag in text.split("|")] for text in self.texts]

        if pad_tokens is not None:
            for pad_token in pad_tokens:
                pad_token_ids = text_model.tokenizer([pad_token]).input_ids[0]
                assert len(pad_token_ids) == 3, f"pad_token:{pad_token} is not a single token"
        self.pad_tokens = pad_tokens

        self.text_infos = []
        for source, target in self.text_pairs:
            src_tokens_ids, tgt_tokens_ids = text_model.tokenizer([source, target], add_special_tokens=False).input_ids
            self.text_infos.append({
                "source": source,
                "target": target,
                "source_length": len(src_tokens_ids),
                "target_length": len(tgt_tokens_ids),
            })

        if selected_tags is not None:
            with open(selected_tags, 'r') as f:
                self.selected_tags = f.read().splitlines()

        self.text_infos = self.text_infos * repeat
        
        self.create_batch()

    def create_batch(self):
        if self.shuffle:
            random.shuffle(self.text_infos)
        self.batch = []
        for i in range(0, len(self.text_infos), self.batch_size):
            self.batch.append(self.text_infos[i:i+self.batch_size])
        return

    def __len__(self):
        return len(self.batch)
    
    def __getitem__(self, idx):
        if idx == 0:
            self.create_batch()
        batch = self.batch[idx]
        sources = []
        targets = []
        for dic in batch:
            source = dic["source"]
            target = dic["target"]
            source_length = dic["source_length"]
            target_length = dic["target_length"]

            add_tags = random.sample(self.selected_tags, random.randint(0, self.max_add_length))
            add_tags = ", ".join(add_tags)
            
            padding = random.choices(self.pad_tokens, k=abs(source_length - target_length))

            if source_length > target_length:
                target += " " + " ".join(padding)
            elif source_length < target_length:
                source += " " + " ".join(padding)

            last_source = self.prefix + ", " if self.prefix else ""
            last_target = self.prefix + ", " if self.prefix else ""
            if random.random() > self.drop_rate:
                if random.random() > self.position_rate:
                    last_source += source + ", " + add_tags
                    last_target += target + ", " + add_tags
                else:
                    last_source += add_tags + ", " + source
                    last_target += add_tags + ", " + target
            else:
                last_source += add_tags
                last_target += add_tags
            
            sources.append(last_source)
            targets.append(last_target)

        return {
            "source": sources,
            "target": targets
        }
            
    

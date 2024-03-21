import torch
from modules.trainer import BaseTrainer

class TLECOTrainer(BaseTrainer):
    def loss(self, batch):
        source = batch["source"]
        target = batch["target"]
        self.batch_size = len(source)

        with torch.autocast("cuda", dtype=self.autocast_dtype):
            with torch.no_grad(), self.network.set_temporary_multiplier(0.0):
                tgt_hidden, tgt_pool = self.text_model(target)
                
            src_hidden, src_pool = self.text_model(source)

            if len(self.network.unet_modules) > 0:
                with torch.no_grad(), self.network.set_temporary_multiplier(0.0):
                    target_kvs = self.kv_emb(tgt_hidden)
                source_kvs = self.kv_emb(src_hidden)

        loss_hidden = torch.nn.functional.mse_loss(src_hidden, tgt_hidden)
        loss_pool = torch.nn.functional.mse_loss(src_pool, tgt_pool)

        loss = loss_hidden + loss_pool

        if len(self.network.unet_modules) > 0:
            loss_kvs =[torch.nn.functional.mse_loss(src, tgt) for src, tgt in zip(source_kvs, target_kvs)]
            for loss_kv in loss_kvs:
                loss += loss_kv / len(loss_kvs)
        
        return loss
    
    def kv_emb(self, text_emb):
        outputs = []
        for name, module in self.diffusion.unet.named_modules():
            if "attn2.to_k" in name:
                outputs.append(module(text_emb))
            elif "attn2.to_v" in name:
                outputs.append(module(text_emb))
        return outputs

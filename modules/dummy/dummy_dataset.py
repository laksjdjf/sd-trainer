from torch.utils.data import Dataset
import torch

class DummyDataset(Dataset):
    def __init__(
        self,
        text_model,
        batch_size = 1,
        size = (512, 512),
        num_batch = 100,
        cache_latent = False,
        cache_text_emb = False,
    ):
        self.batch_size = batch_size
        self.width, self.height = size
        self.num_batch = num_batch
        self.cache_latent = cache_latent
        self.cache_text_emb = cache_text_emb

        self.sdxl = text_model.sdxl 

    def __len__(self):
        return self.num_batch

    def __getitem__(self, i):

        batch = {}
        if self.cache_latent:
            batch["latents"] = torch.randn(self.batch_size, 4, self.height//8, self.width//8)
        else:
            batch["images"] = torch.randn(self.batch_size, 3, self.height, self.width)

        if self.sdxl:
            size_list = [self.height, self.width, 0, 0, self.height, self.width]
            batch["size_condition"] = torch.tensor(size_list).repeat(self.batch_size, 1)

        if self.cache_text_emb:
            dim = 2048 if self.sdxl else 768 # sd2? siranai ko desu ne
            batch["encoder_hidden_states"] = torch.randn(self.batch_size, 77, dim)
            if self.sdxl:
                batch["pooled_outputs"] = torch.randn(self.batch_size, dim)
        else:
            batch["captions"] = ["" for _ in range(self.batch_size)]

        return batch
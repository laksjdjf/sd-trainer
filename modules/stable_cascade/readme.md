うまくいっていない生成だけ。

```python
from modules.stable_cascade.trainer import CascadeTrainer
import torch

trainer = CascadeTrainer.from_pretrained("stabilityai/stable-cascade-prior", "tmp/effnet_encoder.safetensors", "tmp/previewer.safetensors")
trainer.to(dtype=torch.bfloat16)

prompt = "Anthropomorphic cat dressed as a pilot"
negative_prompt = ""
batch_size = 4
height, width = 1024, 1024
guidance_scale = 4.0
step = 30
seed = 4545
images = trainer.sample(prompt, negative_prompt, batch_size, height, width, step, guidance_scale, seed=seed, denoise=1.0)
```

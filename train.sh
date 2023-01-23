#!/bin/bash

python3 main.py \
--model "hakurei/waifu-diffusion" \
--dataset "train_inputs" \
--output "test_model" \
--image_log "image_logs" \
--resolution "512,512" \
--batch_size 4 \
--lr 5e-6 \
--epochs 20 \
--save_n_epochs 5 \
--amp
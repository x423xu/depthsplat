#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=6,7,8,9 python -m src.main +experiment=re10k dataset.roots=[/data0/xxy/data/re10k] data_loader.train.batch_size=8 dataset.test_chunk_interval=100 trainer.max_steps=300000 model.encoder.upsample_factor=4 model.encoder.lowest_feature_resolution=4 checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth output_dir=checkpoints/re10k-256x256-depthsplat-small
# small model
# train on 4x GPUs (>=80GB VRAM) for 150K steps, batch size 8 on each gpu 
python -m src.main +experiment=re10k \
data_loader.train.batch_size=8 \
dataset.test_chunk_interval=10 \
trainer.max_steps=150000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=checkpoints/re10k-256x256-depthsplat-small


# or
# 4x 4090 (24GB) for 300K steps, batch size 4 on each gpu
# python -m src.main +experiment=re10k \
# data_loader.train.batch_size=4 \
# dataset.test_chunk_interval=10 \
# trainer.max_steps=300000 \
# model.encoder.upsample_factor=4 \
# model.encoder.lowest_feature_resolution=4 \
# checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
# checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
# output_dir=checkpoints/re10k-256x256-depthsplat-small


# or
# a single A100 (80GB) for 600K steps, batch size 8 on each gpu
# python -m src.main +experiment=re10k \
# data_loader.train.batch_size=8 \
# dataset.test_chunk_interval=10 \
# trainer.max_steps=600000 \
# model.encoder.upsample_factor=4 \
# model.encoder.lowest_feature_resolution=4 \
# checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
# checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
# output_dir=checkpoints/re10k-256x256-depthsplat-small


# how to resume if training crashes unexpectedly:
# `checkpointing.resume=true`: find latest checkpoint and resume from it
# `wandb.id=WANDB_ID`: continue logging to the same wandb run using the specified WANDB_ID
# python -m src.main +experiment=re10k \
# data_loader.train.batch_size=8 \
# dataset.test_chunk_interval=10 \
# trainer.max_steps=150000 \
# model.encoder.upsample_factor=4 \
# model.encoder.lowest_feature_resolution=4 \
# checkpointing.resume=true \
# wandb.id=WANDB_ID \
# output_dir=checkpoints/re10k-256x256-depthsplat-small


# base model
# train on 4x GPUs (>=80GB VRAM) for 150K steps, batch size 8 on each gpu 
python -m src.main +experiment=re10k \
data_loader.train.batch_size=8 \
dataset.test_chunk_interval=10 \
trainer.max_steps=150000 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vitb.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=checkpoints/re10k-256x256-depthsplat-base


# large model
# train on 4x GPUs (>=80GB VRAM) for 150K steps, batch size 8 on each gpu 
python -m src.main +experiment=re10k \
data_loader.train.batch_size=8 \
dataset.test_chunk_interval=10 \
trainer.max_steps=150000 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitl \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vitl.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=checkpoints/re10k-256x256-depthsplat-large


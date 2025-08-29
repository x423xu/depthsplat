# CUDA_VISIBLE_DEVICES=6 nohup python -m src.main +experiment=re10k dataset.roots=[/data0/xxy/data/re10k] data_loader.train.batch_size=2 dataset.test_chunk_interval=100 trainer.max_steps=100000 model.encoder.upsample_factor=4 model.encoder.lowest_feature_resolution=4 model.encoder.gaussians_per_cell=1 model.encoder.cell_scale=1 model.encoder.down_strides=[2,1] checkpointing.pretrained_model=/data0/xxy/code/depthsplat/checkpoints/re10k-256x256-depthsplat-small/checkpoints/epoch_27-step_300000.ckpt checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth output_dir=checkpoints/gscube_depth21_gpc1_scale1 train_controller.gs_cube=true train_controller.base_model=false >log_depth21_gpc1_scale1 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python -m src.main +experiment=re10k dataset.roots=[/data0/xxy/data/re10k] data_loader.train.batch_size=2 dataset.test_chunk_interval=100 trainer.max_steps=100000 model.encoder.upsample_factor=4 model.encoder.lowest_feature_resolution=4 model.encoder.gaussians_per_cell=1 model.encoder.cell_scale=1 model.encoder.down_strides=[2,2] checkpointing.pretrained_model=/data0/xxy/code/depthsplat/checkpoints/re10k-256x256-depthsplat-small/checkpoints/epoch_27-step_300000.ckpt checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth output_dir=checkpoints/gscube_depth22_gpc1_scale1 train_controller.gs_cube=true train_controller.base_model=false >log_depth22_gpc1_scale1 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup python -m src.main +experiment=re10k \
# dataset.roots=[/data0/xxy/data/re10k] \
# data_loader.train.batch_size=4 \
# dataset.test_chunk_interval=100 \
# trainer.max_steps=100000 \
# model.encoder.upsample_factor=4 \
# model.encoder.lowest_feature_resolution=4 \
# model.encoder.gaussians_per_cell=1 \
# model.encoder.cell_scale=2 \
# model.encoder.down_strides=[3,2] \
# checkpointing.pretrained_model=/data0/xxy/code/depthsplat/checkpoints/re10k-256x256-depthsplat-small/checkpoints/epoch_27-step_300000.ckpt \
# checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
# checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
# output_dir=checkpoints/gscube-depth22-gpc1-scale2-with-skip-large \
# train_controller.gs_cube=true \
# train_controller.base_model=false \
# trainer.val_check_interval=0.1 >gscube_depth22_gpc1_scale2_with_skip_large 2>&1 &

CUDA_VISIBLE_DEVICES=8 nohup python -m src.main +experiment=re10k \
dataset.roots=[/data0/xxy/data/re10k] \
data_loader.train.batch_size=2 \
dataset.test_chunk_interval=100 \
trainer.max_steps=100000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussians_per_cell=1 \
model.encoder.cell_scale=4 \
model.encoder.down_strides=[3,2] \
checkpointing.pretrained_model=/data0/xxy/code/depthsplat/checkpoints/re10k-256x256-depthsplat-small/checkpoints/epoch_27-step_300000.ckpt \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=checkpoints/gscube-depth22-gpc1-scale4-with-skip-large \
train_controller.gs_cube=true \
train_controller.base_model=false \
trainer.val_check_interval=0.25 >gscube_depth22_gpc1_scale4_with_skip_large 2>&1 &
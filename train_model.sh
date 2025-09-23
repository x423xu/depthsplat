CUDA_VISIBLE_DEVICES=9 nohup python -m src.main +experiment=re10k \
dataset.roots=[/data0/xxy/data/re10k] \
data_loader.train.batch_size=2 \
dataset.test_chunk_interval=100 \
trainer.max_steps=100000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussians_per_cell=1 \
model.encoder.cell_scale=2 \
model.encoder.down_strides=[2,2] \
model.encoder.cube_encoder_type=large \
checkpointing.pretrained_model=/data0/xxy/code/depthsplat/checkpoints/re10k-256x256-depthsplat-small/checkpoints/epoch_27-step_300000.ckpt \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=/data2/xxy/code/voxelsplat/checkpoints/gscube-depth22-gpc1-scale2-with-skip-large \
train_controller.gs_cube=true \
train_controller.base_model=false \
trainer.val_check_interval=0.1 \
optimizer.weight_decay=0.0 \
optimizer.lr_gs_cube=2e-4 \
optimizer.lr=1e-7 >gacube_depth22_gpc1_scale2_with_skip_large 2>&1 &

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=32 nohup python -m src.main +experiment=re10k \
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
output_dir=checkpoints/gscube-depth22-gpc1-scale4-with-skip-large-v2 \
train_controller.gs_cube=true \
train_controller.base_model=false \
trainer.val_check_interval=0.25 >gscube_depth22_gpc1_scale4_with_skip_large_v2 2>&1 &

CUDA_VISIBLE_DEVICES=5,7,8,9 OMP_NUM_THREADS=32 nohup python -m src.main +experiment=re10k \
dataset.roots=[/data0/xxy/data/re10k] \
data_loader.train.batch_size=2 \
dataset.test_chunk_interval=100 \
trainer.max_steps=300000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussians_per_cell=1 \
model.encoder.cell_scale=4 \
model.encoder.down_strides=[3,2] \
model.encoder.cube_encoder_type=small \
checkpointing.pretrained_model=/data0/xxy/code/depthsplat/checkpoints/re10k-256x256-depthsplat-small/checkpoints/epoch_27-step_300000.ckpt \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=checkpoints/gscube-depth22-gpc1-scale4-with-skip-small \
train_controller.gs_cube=true \
train_controller.base_model=false \
trainer.val_check_interval=0.05 >gscube_depth22_gpc1_scale4_with_skip_small 2>&1 &


CUDA_VISIBLE_DEVICES=5,7,8,9 OMP_NUM_THREADS=32 nohup python -m src.main +experiment=re10k \
dataset.roots=[/data0/xxy/data/re10k] \
data_loader.train.batch_size=2 \
dataset.test_chunk_interval=100 \
trainer.max_steps=300000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussians_per_cell=1 \
model.encoder.cell_scale=4 \
model.encoder.down_strides=[3,2] \
model.encoder.cube_encoder_type=small \
checkpointing.pretrained_model=/data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small/checkpoints/epoch_1-step_50000.ckpt \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-continue \
train_controller.gs_cube=true \
train_controller.base_model=false \
dataset.view_sampler.warm_up_steps=0 \
trainer.val_check_interval=0.05 >gscube_depth22_gpc1_scale4_with_skip_small_continue 2>&1 &



CUDA_VISIBLE_DEVICES=5,7,8,9 OMP_NUM_THREADS=32 nohup python -m src.main +experiment=re10k \
dataset.roots=[/data0/xxy/data/re10k] \
data_loader.train.batch_size=2 \
dataset.test_chunk_interval=100 \
trainer.max_steps=50000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussians_per_cell=1 \
model.encoder.cell_scale=4 \
model.encoder.down_strides=[3,2] \
model.encoder.cube_encoder_type=small \
checkpointing.pretrained_model=/data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-bf16/checkpoints/epoch_0-step_10000.ckpt \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-bf16 \
train_controller.gs_cube=true \
train_controller.base_model=false \
trainer.val_check_interval=0.05 >gscube_depth22_gpc1_scale4_with_skip_small_bf16 2>&1 &

CUDA_VISIBLE_DEVICES=3,4 OMP_NUM_THREADS=32 nohup python -m src.main +experiment=re10k \
dataset.roots=[/data0/xxy/data/re10k] \
data_loader.train.batch_size=2 \
dataset.test_chunk_interval=100 \
dataset.view_sampler.warm_up_steps=10000 \
trainer.max_steps=300000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussians_per_cell=1 \
model.encoder.cell_scale=4 \
model.encoder.down_strides=[3,2] \
model.encoder.cube_encoder_type=small \
checkpointing.pretrained_model=/data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small/checkpoints/epoch_1-step_50000.ckpt \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-continue \
train_controller.gs_cube=true \
train_controller.base_model=true \
trainer.val_check_interval=0.05 \
optimizer.lr_gs_cube=1e-5 \
optimizer.lr=1e-5 \
optimizer.weight_decay=0.0 \
optimizer.lr_monodepth=1e-5 >gscube_depth22_gpc1_scale4_with_skip_small_continue 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1,2,3,4 OMP_NUM_THREADS=32 nohup python -m src.main +experiment=re10k \
dataset.roots=[/data0/xxy/data/re10k] \
data_loader.train.batch_size=2 \
dataset.test_chunk_interval=100 \
dataset.view_sampler.warm_up_steps=10000 \
trainer.max_steps=300000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussians_per_cell=1 \
model.encoder.cell_scale=2 \
model.encoder.down_strides=[3,2] \
model.encoder.cube_encoder_type=small \
checkpointing.pretrained_model=/data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small/checkpoints/epoch_1-step_50000.ckpt \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=checkpoints/gscube-depth22-gpc1-scale2-with-skip-small-continue \
train_controller.gs_cube=true \
train_controller.base_model=true \
trainer.val_check_interval=0.05 \
optimizer.lr_gs_cube=1e-5 \
optimizer.lr=1e-6 \
optimizer.weight_decay=0.0 \
optimizer.lr_monodepth=1e-7 >gscube_depth22_gpc1_scale2_with_skip_small_continue 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=3,4 OMP_NUM_THREADS=32 nohup python -m src.main +experiment=re10k \
dataset.roots=[/data0/xxy/data/re10k] \
data_loader.train.batch_size=2 \
dataset.test_chunk_interval=100 \
dataset.view_sampler.warm_up_steps=10000 \
trainer.max_steps=300000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussians_per_cell=1 \
model.encoder.cell_scale=4 \
model.encoder.down_strides=[3,2] \
model.encoder.cube_encoder_type=small \
model.encoder.cube_merge_type=mean \
checkpointing.pretrained_model=/data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small/checkpoints/epoch_1-step_50000.ckpt \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-continue-mean \
train_controller.gs_cube=true \
train_controller.base_model=true \
trainer.val_check_interval=0.05 \
optimizer.lr_gs_cube=1e-5 \
optimizer.lr=1e-5 \
optimizer.lr_monodepth=1e-5 >gscube_depth22_gpc1_scale4_with_skip_small_continue_mean 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=16 nohup python -m src.main +experiment=re10k \
dataset.roots=[/data0/xxy/data/re10k] \
data_loader.train.batch_size=2 \
dataset.test_chunk_interval=100 \
trainer.max_steps=75000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussians_per_cell=1 \
model.encoder.cell_scale=4 \
model.encoder.down_strides=[3,2] \
model.encoder.cube_encoder_type=large \
checkpointing.pretrained_model=/data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-large-continue/checkpoints/epoch_0-step_10000.ckpt \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=checkpoints/gscube-depth22-gpc1-scale4-with-skip-large-continue \
train_controller.gs_cube=true \
train_controller.base_model=true \
trainer.val_check_interval=0.1 \
optimizer.lr_gs_cube=1e-6 \
optimizer.lr=1e-6 \
optimizer.lr_monodepth=1e-6 >gscube_depth22_gpc1_scale4_with_skip_large_continue 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=16 python -m src.main +experiment=re10k \
dataset.roots=[/data0/xxy/data/re10k] \
data_loader.train.batch_size=2 \
dataset.test_chunk_interval=100 \
wandb.mode=disabled \
trainer.max_steps=100000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussians_per_cell=1 \
model.encoder.cell_scale=4 \
model.encoder.down_strides=[3,2] \
model.encoder.cube_encoder_type=large \
checkpointing.pretrained_model=/data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-large-continue/checkpoints/epoch_0-step_10000.ckpt \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=checkpoints/gscube-depth22-gpc1-scale4-with-skip-large-continue \
train_controller.gs_cube=true \
train_controller.base_model=true \
trainer.val_check_interval=0.1 \
optimizer.lr_gs_cube=1e-5 \
optimizer.lr=1e-7 \
optimizer.lr_monodepth=1e-6

CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=16 nohup python -m src.main +experiment=re10k_unbounded \
                dataset.roots=[/data0/xxy/data/re10k] \
                data_loader.train.batch_size=4 \
                dataset.test_chunk_interval=10 \
                trainer.max_steps=300000 \
                model.encoder.upsample_factor=4 \
                model.encoder.lowest_feature_resolution=4 \
                checkpointing.pretrained_model=/data0/xxy/code/depthsplat/checkpoints/re10k-256x256-depthsplat-small/checkpoints_backups/epoch_15-step_176096.ckpt \
                checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
                checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
                output_dir=checkpoints/depthsplat_re10k_unbounded_2v >depthsplat_re10k_unbounded_2v 2>&1 &

CUDA_VISIBLE_DEVICES=6,7 nohup python -m src.main +experiment=re10k \
dataset.roots=[/data0/xxy/data/re10k] \
data_loader.train.batch_size=2 \
dataset.test_chunk_interval=100 \
trainer.max_steps=100000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussians_per_cell=1 \
model.encoder.cell_scale=4 \
model.encoder.down_strides=[3,2] \
model.encoder.cube_encoder_type=large \
checkpointing.pretrained_model=/data0/xxy/code/depthsplat/checkpoints/re10k-256x256-depthsplat-small/checkpoints/epoch_27-step_300000.ckpt \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=/data2/xxy/code/voxelsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-large-pa \
train_controller.gs_cube=true \
train_controller.base_model=false \
train_controller.position_aware=true \
trainer.val_check_interval=0.25 >gscube_depth22_gpc1_scale4_with_skip_large_pa 2>&1 &


CUDA_VISIBLE_DEVICES=3,4 OMP_NUM_THREADS=32 nohup python -m src.main +experiment=re10k \
dataset.roots=[/data0/xxy/data/re10k] \
data_loader.train.batch_size=2 \
dataset.test_chunk_interval=100 \
trainer.max_steps=50000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussians_per_cell=1 \
model.encoder.cell_scale=4 \
model.encoder.down_strides=[3,2] \
model.encoder.cube_encoder_type=tiny \
checkpointing.pretrained_model=/data0/xxy/code/depthsplat/checkpoints/re10k-256x256-depthsplat-small/checkpoints/epoch_27-step_300000.ckpt \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=checkpoints/gscube-depth22-gpc1-scale4-with-skip-tiny \
train_controller.gs_cube=true \
train_controller.base_model=false \
trainer.val_check_interval=0.05 >gscube_depth22_gpc1_scale4_with_skip_tiny 2>&1 &

CUDA_VISIBLE_DEVICES=3,4 OMP_NUM_THREADS=32 nohup python -m src.main +experiment=re10k \
dataset.roots=[/data0/xxy/data/re10k] \
data_loader.train.batch_size=2 \
dataset.test_chunk_interval=100 \
dataset.view_sampler.warm_up_steps=30000 \
trainer.max_steps=300000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussians_per_cell=1 \
model.encoder.cell_scale=4 \
model.encoder.down_strides=[3,2] \
model.encoder.cube_encoder_type=tiny \
model.encoder.cube_merge_type=max_conf \
checkpointing.pretrained_model=/data0/xxy/code/depthsplat/checkpoints/re10k-256x256-depthsplat-small/checkpoints/epoch_27-step_300000.ckpt \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=checkpoints/gscube-depth22-gpc1-scale4-with-skip-tiny \
train_controller.gs_cube=true \
train_controller.base_model=true \
trainer.val_check_interval=0.05 \
optimizer.lr=1e-4 >gscube_depth22_gpc1_scale4_with_skip_tiny 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3,4 OMP_NUM_THREADS=32 nohup python -m src.main +experiment=re10k \
dataset.roots=[/data0/xxy/data/re10k] \
data_loader.train.batch_size=2 \
dataset.test_chunk_interval=100 \
dataset.view_sampler.warm_up_steps=30000 \
trainer.max_steps=50000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussians_per_cell=1 \
model.encoder.cell_scale=4 \
model.encoder.down_strides=[3,2] \
model.encoder.cube_encoder_type=small_v2 \
model.encoder.cube_merge_type=max_conf \
checkpointing.pretrained_model=/data0/xxy/code/depthsplat/checkpoints/re10k-256x256-depthsplat-small/checkpoints/epoch_27-step_300000.ckpt \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=checkpoints/gscube-depth22-gpc1-scale4-with-skip-small_v2 \
train_controller.gs_cube=true \
train_controller.base_model=false \
trainer.val_check_interval=0.05 \
optimizer.lr=1e-4 >gscube_depth22_gpc1_scale4_with_skip_small_v2 2>&1 < /dev/null &


# train small v2 continue
CUDA_VISIBLE_DEVICES=1,2,3,4 OMP_NUM_THREADS=32 nohup python -m src.main +experiment=re10k \
dataset.roots=[/data0/xxy/data/re10k] \
data_loader.train.batch_size=2 \
dataset.test_chunk_interval=100 \
dataset.view_sampler.warm_up_steps=0 \
trainer.max_steps=300000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussians_per_cell=1 \
model.encoder.cell_scale=4 \
model.encoder.down_strides=[3,2] \
model.encoder.cube_encoder_type=small_v2 \
model.encoder.cube_merge_type=max_conf \
checkpointing.pretrained_model=/data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small_v2/checkpoints/epoch_1-step_50000.ckpt \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-v2-continue \
train_controller.gs_cube=true \
train_controller.base_model=true \
trainer.val_check_interval=0.05 \
optimizer.lr_gs_cube=1e-5 \
optimizer.lr=1e-5 \
optimizer.lr_monodepth=1e-5 >gscube_depth22_gpc1_scale4_with_skip_small_v2_continue 2>&1 < /dev/null &

# resume training small v2 continue
CUDA_VISIBLE_DEVICES=1,2,3,4 OMP_NUM_THREADS=32 python -m src.main +experiment=re10k \
dataset.roots=[/data0/xxy/data/re10k] \
data_loader.train.batch_size=2 \
dataset.test_chunk_interval=100 \
dataset.view_sampler.warm_up_steps=0 \
trainer.max_steps=300000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussians_per_cell=1 \
model.encoder.cell_scale=4 \
model.encoder.down_strides=[3,2] \
model.encoder.cube_encoder_type=small_v2 \
model.encoder.cube_merge_type=max_conf \
checkpointing.pretrained_model=/data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small_v2/checkpoints/epoch_1-step_50000.ckpt \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-v2-continue \
train_controller.gs_cube=true \
train_controller.base_model=true \
trainer.val_check_interval=0.05 \
optimizer.lr_gs_cube=1e-5 \
optimizer.lr=1e-5 \
optimizer.lr_monodepth=1e-5 \
checkpointing.resume=true
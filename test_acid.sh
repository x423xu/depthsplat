CUDA_VISIBLE_DEVICES=1 python -m src.main +experiment=re10k \
mode=test \
dataset.roots=[/data2/xxy/data/acid] \
dataset.view_sampler.index_path=assets/evaluation_index_acid_4v.json \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
checkpointing.pretrained_model=/data0/xxy/code/depthsplat/checkpoints/re10k-256x256-depthsplat-small/checkpoints/epoch_27-step_300000.ckpt \
test.save_video=false \
test.save_gaussian=false \
test.compute_scores=true \
test.render_chunk_size=10 \
output_dir=notes/depthsplat_acid_4v

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
mode=test \
dataset.roots=[/data2/xxy/data/acid] \
dataset.view_sampler.index_path=assets/evaluation_index_acid_4v.json \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=4 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussians_per_cell=1 \
model.encoder.cell_scale=4 \
model.encoder.down_strides=[3,2] \
checkpointing.pretrained_model=/data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-maxconf-scratch-4w/checkpoints/epoch_11-step_380000.ckpt \
train_controller.gs_cube=true \
test.save_video=false \
test.save_gaussian=false \
test.compute_scores=true \
test.render_chunk_size=10 \
output_dir=notes/gscube_acid_4v \
model.encoder.cube_encoder_type=small \
model.encoder.cube_merge_type=mean

CUDA_VISIBLE_DEVICES=2 python -m src.main +experiment=re10k \
mode=test \
dataset.roots=[/data2/xxy/data/acid] \
dataset.view_sampler.index_path=assets/evaluation_index_acid_4v.json \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
checkpointing.pretrained_model=/data2/xxy/code/depthsplat/checkpoints/depthsplat-anysplat/checkpoints/epoch_3-step_100000-v1.ckpt \
test.save_video=false \
test.save_gaussian=false \
test.compute_scores=true \
test.render_chunk_size=10 \
output_dir=notes/depthsplat_acid_4v \
train_controller.gs_cube=false \
train_controller.gaussian_merge=true
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
mode=test \
dataset.roots=[/data2/xxy/data/dl3dv] \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=6 \
dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth
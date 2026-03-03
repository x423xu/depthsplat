CUDA_VISIBLE_DEVICES=2 python -m notes.online_test --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_2v_0.8ind.json --out_dir notes/gscube_2v
CUDA_VISIBLE_DEVICES=2 python -m notes.online_test --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_4v_0.8ind.json --out_dir notes/gscube_4v
CUDA_VISIBLE_DEVICES=2 python -m notes.online_test --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_6v_0.8ind.json --out_dir notes/gscube_6v
CUDA_VISIBLE_DEVICES=2 python -m notes.online_test --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_8v_0.8ind.json --out_dir notes/gscube_8v --iter_depth --anchor_features


#bounded
CUDA_VISIBLE_DEVICES=3 python -m notes.online_test --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k.json --out_dir notes/gscube_2v_bounded --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-maxconf-scratch-4w/checkpoints_backups/epoch_10-step_346670.ckpt

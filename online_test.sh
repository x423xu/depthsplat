# max gscube nov=4, bs=2, ds=x
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test  --batch_forward --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-continue/checkpoints_backups/epoch_8-step_295488.ckpt --cube_merge_type max --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_4v.json
# max gscube nov=6, bs=2, ds=x
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test  --batch_forward --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-continue/checkpoints_backups/epoch_8-step_295488.ckpt --cube_merge_type max --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_6v.json
# max gscube nov=8, bs=2, ds=x
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test  --batch_forward --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-continue/checkpoints_backups/epoch_8-step_295488.ckpt --cube_merge_type max --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_8v.json

# mean gscube nov=4, bs=2, ds=x
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test  --batch_forward --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-maxconf-scratch-4w/checkpoints_backups/epoch_1-step_66018.ckpt --cube_merge_type mean --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_4v.json
# mean gscube nov=6, bs=2, ds=x
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test  --batch_forward --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-maxconf-scratch-4w/checkpoints_backups/epoch_1-step_66018.ckpt --cube_merge_type mean --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_6v.json
# mean gscube nov=8, bs=2, ds=x
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test  --batch_forward --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-maxconf-scratch-4w/checkpoints_backups/epoch_1-step_66018.ckpt --cube_merge_type mean --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_8v.json
# mean gscube nov=4, bs=x, ds=2
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test --iter_depth --view_base=2  --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-maxconf-scratch-4w/checkpoints_backups/epoch_1-step_66018.ckpt --cube_merge_type mean --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_4v.json
# mean gscube nov=6, bs=x, ds=2
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test --iter_depth --view_base=2  --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-maxconf-scratch-4w/checkpoints_backups/epoch_1-step_66018.ckpt --cube_merge_type mean --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_6v.json
# mean gscube nov=8, bs=x, ds=2
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test --iter_depth --view_base=2  --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-maxconf-scratch-4w/checkpoints_backups/epoch_1-step_66018.ckpt --cube_merge_type mean --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_8v.json
# mean gscube nov=4, bs=x, ds=x
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test   --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-maxconf-scratch-4w/checkpoints_backups/epoch_1-step_66018.ckpt --cube_merge_type mean --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_4v.json
# mean gscube nov=6, bs=x, ds=x
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test   --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-maxconf-scratch-4w/checkpoints_backups/epoch_1-step_66018.ckpt --cube_merge_type mean --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_6v.json
# mean gscube nov=8, bs=x, ds=x
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test   --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-maxconf-scratch-4w/checkpoints_backups/epoch_1-step_66018.ckpt --cube_merge_type mean --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_8v.json

# maxconf gscube nov=4, bs=2, ds=x
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test  --batch_forward --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-continue/checkpoints_backups/epoch_8-step_295488.ckpt --cube_merge_type max_conf --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_4v.json
# maxconf gscube nov=6, bs=2, ds=x
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test  --batch_forward --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-continue/checkpoints_backups/epoch_8-step_295488.ckpt --cube_merge_type max_conf --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_6v.json
# maxconf gscube nov=8, bs=2, ds=x
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test  --batch_forward --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-continue/checkpoints_backups/epoch_8-step_295488.ckpt --cube_merge_type max_conf --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_8v.json
# maxconf gscube nov=4, bs=x, ds=2
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test --iter_depth --view_base=2  --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-continue/checkpoints_backups/epoch_8-step_295488.ckpt --cube_merge_type max_conf --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_4v.json
# maxconf gscube nov=6, bs=x, ds=2
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test --iter_depth --view_base=2  --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-continue/checkpoints_backups/epoch_8-step_295488.ckpt --cube_merge_type max_conf --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_6v.json
# maxconf gscube nov=8, bs=x, ds=2
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test --iter_depth --view_base=2  --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-continue/checkpoints_backups/epoch_8-step_295488.ckpt --cube_merge_type max_conf --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_8v.json
# maxconf gscube nov=4, bs=x, ds=x
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test   --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-continue/checkpoints_backups/epoch_8-step_295488.ckpt --cube_merge_type max_conf --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_4v.json
# maxconf gscube nov=6, bs=x, ds=x
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test   --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-continue/checkpoints_backups/epoch_8-step_295488.ckpt --cube_merge_type max_conf --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_6v.json
# maxconf gscube nov=8, bs=x, ds=x
CUDA_VISIBLE_DEVICES=9 python -m notes.online_test   --chunk_num=2  --model_path /data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-continue/checkpoints_backups/epoch_8-step_295488.ckpt --cube_merge_type max_conf --model gscube --index_path /data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_8v.json

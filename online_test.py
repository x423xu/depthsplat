'''
Online test for context views, 4,6,8,10,12,20,30,40
Author: xxy
Date: 2024/06/20
'''


'''
Notes: we need to understand why more contexts work worse, and then improve it.
Reason I guess:
   1. more context views bring unexpected points. Then our voxelizaition can remove some. Then will it stills fail? if yes, try to only use anchor points.
'''

import os,sys
os.environ["OMP_NUM_THREADS"] = "32"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
sys.path.append("/data2/xxy/code/depthsplat/")
sys.path.append("/data2/xxy/code/depthsplat/src")
import hydra
import torch
from hydra.experimental import initialize_config_dir, compose
from omegaconf import OmegaConf
from jaxtyping import install_import_hook
from pytorch_lightning import Trainer
from pathlib import Path
from colorama import Fore
from tqdm.notebook import tqdm
import json

with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.misc.resume_ckpt import find_latest_ckpt
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper

def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

import pytorch_lightning as pl
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
pl.seed_everything(42)
import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, default='depthsplat', choices=['depthsplat', 'gscube', 'depth_any'])
parser.add_argument('--model', type=str, default='gscube', choices=['depthsplat', 'gscube', 'depth_any'])
# parser.add_argument('--model', type=str, default='depth_any', choices=['depthsplat', 'gscube', 'depth_any'])
# parser.add_argument('--index_path', type=str, default='/data2/xxy/code/depthsplat/assets/evaluation_index_re10k_all_8v_0.8ind.json')
parser.add_argument('--index_path', type=str, default='/data0/xxy/code/CanonicalGS/assets/re10k_8v_canonical_len100_indicies.json')
# parser.add_argument('--index_path', type=str, default='/data2/xxy/code/depthsplat/assets/evaluation_index_re10k.json')
# parser.add_argument('--index_path', type=str, default='/data2/xxy/code/depthsplat/assets/re10k_6v_indicies_expansion.json')
parser.add_argument('--view_base', type=int, default=4)
parser.add_argument('--iter_depth', action='store_true', default=False)
parser.add_argument('--batch_forward', action='store_true', default=False)
parser.add_argument('--chunk_num', type=int, default=2)
parser.add_argument('--anchor_features', action='store_true', default=False)
parser.add_argument('--anchor_base', type=int, default=4)
# parser.add_argument('--model_path', type=str, default='/data2/xxy/code/depthsplat/checkpoints/depthsplat-anysplat/checkpoints/epoch_3-step_100000-v1.ckpt')
parser.add_argument('--model_path', type=str, default='/data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-maxconf-scratch-4w/checkpoints_backups/epoch_10-step_340070.ckpt')
# parser.add_argument('--model_path', type=str, default='/data0/xxy/code/depthsplat/checkpoints/re10k-256x256-depthsplat-small/checkpoints/epoch_27-step_300000.ckpt')
parser.add_argument('--cube_merge_type', type=str, default='mean')
# parser.add_argument('--cube_merge_type', type=str, default='max_conf')
# parser.add_argument('--out_dir', type=str, default='notes/tmp_depthsplat_re10k')
# parser.add_argument('--out_dir', type=str, default='notes/tmp_gscube_re10k')
parser.add_argument('--out_dir', type=str, default='notes/tmp')
parser.add_argument('--cell_scale', type=float, default=2.8)
extra_args = parser.parse_args()
cell_scale = extra_args.cell_scale
print(extra_args)
if extra_args.model == 'depthsplat':
    overrides = [
        "+experiment=re10k",
        "dataset.roots=[/data0/xxy/data/re10k]",
        "data_loader.train.batch_size=2",
        "dataset.test_chunk_interval=7000",
        "trainer.max_steps=300000",
        "model.encoder.upsample_factor=4",
        "model.encoder.lowest_feature_resolution=4",
        "model.encoder.gaussians_per_cell=1",
        "model.encoder.cell_scale=4.0",
        "model.encoder.down_strides=[3,2]",
        "checkpointing.pretrained_model=/data0/xxy/code/depthsplat/checkpoints/re10k-256x256-depthsplat-small/checkpoints/epoch_27-step_300000.ckpt",
        "checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth",
        "checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth",
        "output_dir=notes/depthsplat_tmp",
        "train_controller.gs_cube=false",
        "train_controller.base_model=true",
        "trainer.val_check_interval=0.25",
        "dataset/view_sampler=evaluation",
        "dataset.view_sampler.index_path=/data2/xxy/code/depthsplat/assets/evaluation_index_re10k_video.json",
        "dataset.view_sampler.num_context_views=2",
        "mode=test",
        "test.save_video=true",
        "test.save_gaussian=false",
        "test.compute_scores=true",
        # "test.save_image=true",
        # "test.save_gt_image=true",
        "test.render_chunk_size=10",
        "model.encoder.cube_encoder_type=large",
        # "dataset.overfit_to_scene=a47b88040452d7d9",
        # "dataset.overfit_to_scene=de129b4aa11af575"
        # "dataset.overfit_to_scene=9266d12186a3612c",
    ]

if extra_args.model == 'gscube':
    overrides = [
        "+experiment=re10k",
        "dataset.roots=[/data0/xxy/data/re10k]",
        "data_loader.train.batch_size=2",
        "dataset.test_chunk_interval=1000",
        "model.encoder.upsample_factor=4",
        "model.encoder.lowest_feature_resolution=4",
        "model.encoder.gaussians_per_cell=1",
        "model.encoder.cell_scale=4.0",
        "model.encoder.down_strides=[3,2]",
        "checkpointing.pretrained_model=/data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-continue/checkpoints_backups/epoch_8-step_295488.ckpt",
        # "checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth",
        # "checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth",
        "output_dir=notes/tmp",
        "train_controller.gs_cube=true",
        "train_controller.base_model=true",
        "trainer.val_check_interval=0.25",
        "dataset/view_sampler=evaluation",
        "dataset.view_sampler.index_path=/data2/xxy/code/depthsplat/assets/evaluation_index_re10k_video.json",
        "dataset.view_sampler.num_context_views=2",
        "mode=test",
        "test.save_video=false",
        "test.save_gaussian=false",
        "test.compute_scores=true",
        "test.save_image=true",
        "test.save_gt_image=true",
        # "test.save_depth=true",
        # "test.save_depth_concat_img=true",
        "test.render_chunk_size=10",
        "model.encoder.cube_encoder_type=small",
        # "dataset.overfit_to_scene=a47b88040452d7d9",
        # "dataset.overfit_to_scene=de129b4aa11af575",
        # "dataset.overfit_to_scene=9b66fe45498df0ad",
        "model.encoder.cube_merge_type=max",
    ]
if extra_args.model == 'depth_any':
    overrides = [
        "+experiment=re10k",
        "dataset.roots=[/data0/xxy/data/re10k]",
        # "dataset.test_chunk_interval=100",
        "model.encoder.upsample_factor=4",
        "model.encoder.lowest_feature_resolution=4",
        "checkpointing.pretrained_model=/data2/xxy/code/depthsplat/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-continue/checkpoints_backups/epoch_8-step_295488.ckpt",
        "checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth",
        "checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth",
        "output_dir=notes/tmp",
        "train_controller.gs_cube=false",
        "train_controller.base_model=true",
        "trainer.val_check_interval=0.25",
        "dataset/view_sampler=evaluation",
        "dataset.view_sampler.index_path=/data2/xxy/code/depthsplat/assets/evaluation_index_re10k_video.json",
        "dataset.view_sampler.num_context_views=2",
        "mode=test",
        "test.save_video=true",
        "test.save_gaussian=false",
        "test.compute_scores=true",
        # "test.save_depth=true",
        # "test.save_depth_concat_img=true",
        "test.render_chunk_size=10",
        # "dataset.overfit_to_scene=a47b88040452d7d9",
        # "dataset.overfit_to_scene=25cb07fc0e6b4b4e",
        "dataset.overfit_to_scene=9266d12186a3612c",
        "train_controller.gaussian_merge=true"
    ]

new_overrides = []
for o in overrides:
    if o.startswith("dataset.view_sampler.index_path="):
        o = f'dataset.view_sampler.index_path={extra_args.index_path}'
    if o.startswith("checkpointing.pretrained_model="):
        o = f'checkpointing.pretrained_model={extra_args.model_path}'
    if o.startswith("model.encoder.cube_merge_type="):
        o = f'model.encoder.cube_merge_type={extra_args.cube_merge_type}'
    if o.startswith("output_dir="):
        if cell_scale is not None:
            o = f'output_dir={extra_args.out_dir}_scale{cell_scale}'
        else:
            o = f'output_dir={extra_args.out_dir}'
    if o.startswith("model.encoder.cell_scale=") and (cell_scale is not None):
        o = f'model.encoder.cell_scale={cell_scale}'
    new_overrides.append(o)
overrides = new_overrides

torch.set_float32_matmul_precision('high')
with initialize_config_dir(config_dir=os.path.abspath('config')):
    cfg_dict = compose(config_name='main', overrides=overrides)
# print(OmegaConf.to_yaml(cfg_dict))
cfg = load_typed_root_config(cfg_dict)
set_cfg(cfg_dict)
print(cfg_dict.output_dir)
print(cfg.train_controller)
output_dir = Path(cfg_dict.output_dir)
os.makedirs(output_dir, exist_ok=True)

step_tracker = StepTracker()
print(torch.cuda.device_count(), "GPUs available")
trainer = Trainer(
    max_epochs=-1,
    accelerator="gpu",
    devices=torch.cuda.device_count(),
    strategy="auto",
    val_check_interval=cfg.trainer.val_check_interval,
    enable_progress_bar=True,
    gradient_clip_val=cfg.trainer.gradient_clip_val,
    max_steps=cfg.trainer.max_steps,
    num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
    num_nodes=cfg.trainer.num_nodes,
    plugins=None,
    profiler=None,
)

checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

torch.manual_seed(cfg_dict.seed + trainer.global_rank)

encoder, encoder_visualizer = get_encoder(cfg.model.encoder, gs_cube = cfg.train_controller.gs_cube,vggt_meta = cfg.train_controller.vggt_meta,gaussian_merge=cfg.train_controller.gaussian_merge)

model_wrapper = ModelWrapper(
    cfg.optimizer,
    cfg.test,
    cfg.train,
    encoder,
    encoder_visualizer,
    get_decoder(cfg.model.decoder, cfg.dataset),
    get_losses(cfg.loss),
    step_tracker,
    eval_data_cfg=None,
    train_controller_cfg=cfg.train_controller,
    iter_depth=extra_args.iter_depth,
    view_base=extra_args.view_base,
    batch_forward=extra_args.batch_forward,
    anchor_features=extra_args.anchor_features,
    chunk_num=extra_args.chunk_num,
    anchor_base=extra_args.anchor_base
)

data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
        train_controller_cfg=cfg.train_controller,
    )
strict_load = not cfg.checkpointing.no_strict_load

if cfg.checkpointing.pretrained_model is not None:
    pretrained_model = torch.load(cfg.checkpointing.pretrained_model, map_location='cpu')
    if 'state_dict' in pretrained_model:
        pretrained_model = pretrained_model['state_dict']

    model_wrapper.load_state_dict(pretrained_model, strict=strict_load)
    print(
        cyan(
            f"Loaded pretrained weights: {cfg.checkpointing.pretrained_model}"
        )
    )

# load pretrained depth model only
if cfg.checkpointing.pretrained_depth is not None:
    pretrained_model = torch.load(cfg.checkpointing.pretrained_depth, map_location='cpu')['model']

    strict_load = True
    model_wrapper.encoder.depth_predictor.load_state_dict(pretrained_model, strict=strict_load)
    print(
        cyan(
            f"Loaded pretrained depth: {cfg.checkpointing.pretrained_depth}"
        )
    )
   
trainer.test(
    model_wrapper,
    datamodule=data_module,
    ckpt_path=checkpoint_path,
)

# scene_names = {}
# test_loader = data_module.test_dataloader()
# print(len(test_loader.dataset))
# i=0
# for batch in test_loader:
#     scene = batch['scene'][0]
#     print(i, scene)
#     scene_names[i] = scene
#     i+=1
    

# save_path = 'notes/scene_names.json'
# with open(save_path, 'w') as f:
#     json.dump(scene_names, f, indent=4)
# print(f"Saved scene names to {save_path}")

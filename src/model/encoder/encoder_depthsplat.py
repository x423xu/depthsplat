from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from torch import einsum
from einops import repeat,rearrange

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, GaussianCubeAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_depthsplat_cfg import EncoderVisualizerDepthSplatCfg

import torchvision.transforms as T
import torch.nn.functional as F

from .unimatch.mv_unimatch import MultiViewUniMatch
from .unimatch.dpt_head import DPTHead


@dataclass
class EncoderDepthSplatCfg:
    name: Literal["depthsplat"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerDepthSplatCfg
    gaussian_adapter: GaussianAdapterCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]

    # mv_unimatch
    num_scales: int
    upsample_factor: int
    lowest_feature_resolution: int
    depth_unet_channels: int
    grid_sample_disable_cudnn: bool

    # depthsplat color branch
    large_gaussian_head: bool
    color_large_unet: bool
    init_sh_input_img: bool
    feature_upsampler_channels: int
    gaussian_regressor_channels: int

    # loss config
    supervise_intermediate_depth: bool
    return_depth: bool

    # only depth
    train_depth_only: bool

    # monodepth config
    monodepth_vit_type: str

    # multi-view matching
    local_mv_match: int

    gs_cube: bool = False  # whether to use the gs cube model


class EncoderDepthSplat(Encoder[EncoderDepthSplatCfg]):
    def __init__(self, cfg: EncoderDepthSplatCfg) -> None:
        super().__init__(cfg)

        self.depth_predictor = MultiViewUniMatch(
            num_scales=cfg.num_scales,
            upsample_factor=cfg.upsample_factor,
            lowest_feature_resolution=cfg.lowest_feature_resolution,
            vit_type=cfg.monodepth_vit_type,
            unet_channels=cfg.depth_unet_channels,
            grid_sample_disable_cudnn=cfg.grid_sample_disable_cudnn,
        )

        if self.cfg.train_depth_only:
            return

        # upsample features to the original resolution
        model_configs = {
            'vits': {'in_channels': 384, 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'in_channels': 768, 'features': 96, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'in_channels': 1024, 'features': 128, 'out_channels': [128, 256, 512, 1024]},
        }

        self.feature_upsampler = DPTHead(**model_configs[cfg.monodepth_vit_type],
                                        downsample_factor=cfg.upsample_factor,
                                        return_feature=True,
                                        num_scales=cfg.num_scales,
                                        )
        feature_upsampler_channels = model_configs[cfg.monodepth_vit_type]["features"]
        
        # gaussians adapter
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # concat(img, depth, match_prob, features)
        in_channels = 3 + 1 + 1 + feature_upsampler_channels
        channels = self.cfg.gaussian_regressor_channels

        # conv regressor
        modules = [
                    nn.Conv2d(in_channels, channels, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, 3, 1, 1),
                ]

        self.gaussian_regressor = nn.Sequential(*modules)

        # predict gaussian parameters: scale, q, sh, offset, opacity
        num_gaussian_parameters = self.gaussian_adapter.d_in + 2 + 1

        # concat(img, features, regressor_out, match_prob)
        in_channels = 3 + feature_upsampler_channels + channels + 1
        self.gaussian_head = nn.Sequential(
                nn.Conv2d(in_channels, num_gaussian_parameters,
                          3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(num_gaussian_parameters,
                          num_gaussian_parameters, 3, 1, 1, padding_mode='replicate')
            )

        if self.cfg.init_sh_input_img:
            nn.init.zeros_(self.gaussian_head[-1].weight[10:])
            nn.init.zeros_(self.gaussian_head[-1].bias[10:])

        # init scale
        # first 3: opacity, offset_xy
        nn.init.zeros_(self.gaussian_head[-1].weight[3:6])
        nn.init.zeros_(self.gaussian_head[-1].bias[3:6])

        # whether to use the gs cube model
        self.gs_cube = cfg.gs_cube
        if cfg.gs_cube:
            # use the gs cube model
            self.gaussian_head_x = nn.Sequential(
                nn.Conv2d(in_channels, num_gaussian_parameters, 3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(num_gaussian_parameters,num_gaussian_parameters, 3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((64, 1)), # if we want rank=k, just change to (64, k)
                nn.Flatten(-2, -1),
                nn.Linear(64, 64),
            )
            self.gaussian_head_y = nn.Sequential(
                nn.Conv2d(in_channels, num_gaussian_parameters, 3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(num_gaussian_parameters,num_gaussian_parameters, 3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((64, 1)),
                nn.Flatten(-2, -1),
                nn.Linear(64, 64),
            )
            self.gaussian_head_z = nn.Sequential(
                nn.Conv2d(256*256, num_gaussian_parameters, 3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(num_gaussian_parameters,num_gaussian_parameters, 3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.AdaptiveMaxPool2d((128, 1)),
                nn.Flatten(-2, -1),
                nn.Linear(128, 128),
            )
            self.gaussian_cube_adapter = GaussianCubeAdapter(cfg.gaussian_adapter)
            self.coef_module = nn.Sequential(
                nn.Conv2d(in_channels, num_gaussian_parameters, 3, 1, 1, padding_mode='replicate'),
                nn.BatchNorm2d(num_gaussian_parameters),
                nn.GELU(),
                nn.Conv2d(num_gaussian_parameters,num_gaussian_parameters, 3, 1, 1, padding_mode='replicate'),
                nn.BatchNorm2d(num_gaussian_parameters),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(-3, -1),
                nn.Linear(num_gaussian_parameters,1),
                # nn.Tanh(),
            )

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        if v > 3:
            with torch.no_grad():
                xyzs = context["extrinsics"][:, :, :3, -1].detach()
                cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
                cameras_dist_index = torch.argsort(cameras_dist_matrix)

                cameras_dist_index = cameras_dist_index[:, :, :(self.cfg.local_mv_match + 1)]
        else:
            cameras_dist_index = None

        # depth prediction
        results_dict = self.depth_predictor(
            context["image"],
            attn_splits_list=[2],
            min_depth=1. / context["far"],
            max_depth=1. / context["near"],
            intrinsics=context["intrinsics"],
            extrinsics=context["extrinsics"],
            nn_matrix=cameras_dist_index,
        )

        # list of [B, V, H, W], with all the intermediate depths
        depth_preds = results_dict['depth_preds']

        # [B, V, H, W]
        depth = depth_preds[-1]

        if self.cfg.train_depth_only:
            # convert format
            # [B, V, H*W, 1, 1]
            depths = rearrange(depth, "b v h w -> b v (h w) () ()")

            if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
                # supervise all the intermediate depth predictions
                num_depths = len(depth_preds)

                # [B, V, H*W, 1, 1]
                intermediate_depths = torch.cat(
                    depth_preds[:(num_depths - 1)], dim=0)
                intermediate_depths = rearrange(
                    intermediate_depths, "b v h w -> b v (h w) () ()")

                # concat in the batch dim
                depths = torch.cat((intermediate_depths, depths), dim=0)

                b *= num_depths

            # return depth prediction for supervision
            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, V, H, W]

            return {
                "gaussians": None,
                "depths": depths
            }

        # features [BV, C, H, W]
        features = self.feature_upsampler(results_dict["features_mono_intermediate"],
                                          cnn_features=results_dict["features_cnn_all_scales"][::-1],
                                          mv_features=results_dict["features_mv"][
                                          0] if self.cfg.num_scales == 1 else results_dict["features_mv"][::-1]
                                          )

        # match prob from softmax
        # [BV, D, H, W] in feature resolution
        match_prob = results_dict['match_probs'][-1]
        match_prob = torch.max(match_prob, dim=1, keepdim=True)[
            0]  # [BV, 1, H, W]
        match_prob = F.interpolate(
            match_prob, size=depth.shape[-2:], mode='nearest')

        # unet input
        concat = torch.cat((
            rearrange(context["image"], "b v c h w -> (b v) c h w"),
            rearrange(depth, "b v h w -> (b v) () h w"),
            match_prob,
            features,
        ), dim=1)

        out = self.gaussian_regressor(concat)

        concat = [out,
                    rearrange(context["image"],
                            "b v c h w -> (b v) c h w"),
                    features,
                    match_prob]

        out = torch.cat(concat, dim=1)

        gaussians = self.gaussian_head(out)  # [BV, C, H, W]

        gaussians = rearrange(gaussians, "(b v) c h w -> b v c h w", b=b, v=v)

        depths = rearrange(depth, "b v h w -> b v (h w) () ()")

        # get the gs_cube mask, this mask is in camera coordinate, this is not what we want
        # we should calculate it from depths. It is noted that if the cube is scaled, the depth should be scaled as well.
        if self.gs_cube:
            # s1: interpolate to cube size
            depth_cube = F.interpolate(depth.detach(),
                                       size=(64, 64),
                                        mode='bilinear',
                                        align_corners=True,
                                        )
            depth_cube.clamp_(min=0.5, max=100)  # avoid division by zero
            # s2: get inverse depth raning 0,1
            depth_cube_inverse = 1. / depth_cube # b v 64 64
            # mask_init = torch.zeros_like(results_dict['match_probs'][-1]) #bv, 64,64,128
            linear_space = (
                    torch.linspace(0, 1, results_dict['match_probs'][-1].shape[1]+1)
                    .type_as(depth_cube_inverse)
                    .view(1, 1, results_dict['match_probs'][-1].shape[1]+1, 1, 1)
                )  # [1, 1, D, 1, 1]
            linear_start = linear_space[:,:,:-1,...]
            linear_end = linear_space[:,:,1:,...]
            mask = (depth_cube_inverse.unsqueeze(2) >= linear_start) & (
                depth_cube_inverse.unsqueeze(2) < linear_end)  # [b, v, D, 64, 64]
            mask = rearrange(mask, "b v d h w -> b v h w d")
            

        # [B, V, H*W, 1, 1]
        densities = rearrange(
            match_prob, "(b v) c h w -> b v (c h w) () ()", b=b, v=v)
        # [B, V, H*W, 84]
        raw_gaussians = rearrange(
            gaussians, "b v c h w -> b v (h w) c")

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:

            # supervise all the intermediate depth predictions
            num_depths = len(depth_preds)

            # [B, V, H*W, 1, 1]
            intermediate_depths = torch.cat(
                depth_preds[:(num_depths - 1)], dim=0)
            
            intermediate_depths = rearrange(
                intermediate_depths, "b v h w -> b v (h w) () ()")

            # concat in the batch dim
            depths = torch.cat((intermediate_depths, depths), dim=0)

            # shared color head
            densities = torch.cat([densities] * num_depths, dim=0)
            raw_gaussians = torch.cat(
                [raw_gaussians] * num_depths, dim=0)

            b *= num_depths

        # [B, V, H*W, 1, 1]
        opacities = raw_gaussians[..., :1].sigmoid().unsqueeze(-1)
        raw_gaussians = raw_gaussians[..., 1:]
        
        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / \
            torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

        sh_input_images = context["image"]

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
            context_extrinsics = torch.cat(
                [context["extrinsics"]] * len(depth_preds), dim=0)
            context_intrinsics = torch.cat(
                [context["intrinsics"]] * len(depth_preds), dim=0)

            gaussians = self.gaussian_adapter.forward(
                rearrange(context_extrinsics, "b v i j -> b v () () () i j"),
                rearrange(context_intrinsics, "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                opacities,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
                input_images=sh_input_images.repeat(
                    len(depth_preds), 1, 1, 1, 1) if self.cfg.init_sh_input_img else None,
            )


        else:
            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"],
                          "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"],
                          "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                opacities,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
                input_images=sh_input_images if self.cfg.init_sh_input_img else None,
            )

            if self.gs_cube:
                # prepare gs_cube for rendering
                # seperately train gs_cube
                gs_in = out.detach()
                # here a NxHxW tensor is decomposed into three tensors:
                # gaussians_x: [BV, C, 64]
                # gaussians_y: [BV, C, 64]
                # gaussians_z: [BV, C, 128]
                # the compression ratio is 256*256*2 / (64 + 64 + 128) = 512    
                gaussians_x = self.gaussian_head_x(rearrange(gs_in, "n c h w -> n c w h")) # bv, c, 64
                gaussians_y = self.gaussian_head_y(gs_in) # bv, c, 64
                gaussians_z = self.gaussian_head_z(rearrange(gs_in, "n c h w -> n (h w) c ()"))# bv, c, 128
                coef = self.coef_module(gs_in) # bv, c, 1

                raw_gaussian_cube_cp = torch.cat([gaussians_x, gaussians_y, gaussians_z], dim=-1) # bv, c, 64+64+128
                raw_gaussian_cube_cp = rearrange(raw_gaussian_cube_cp, "(b v) c n -> b v n c", b=b, v=v)

                # getting opacities from raw_gaussian_cube_cp
                raw_opacities_cube_cp = raw_gaussian_cube_cp[..., :1].unsqueeze(-1)
                raw_gaussian_cube_cp = raw_gaussian_cube_cp[..., 1:]

                # getting ray in camera coordinate
                xy_ray_cube, _ = sample_image_grid((64, 64), device)
                xy_ray_cube = rearrange(xy_ray_cube, "h w xy -> (h w) () xy")
                gaussians_cube_cp = rearrange(
                    raw_gaussian_cube_cp,
                    "... (srf c) -> ... srf c",
                    srf=self.cfg.num_surfaces,
                ) # b v n srf c
                # offset_xy_cube = gaussians_cube_cp[..., :2].sigmoid()  
                offset_xy_cube = F.interpolate(
                    rearrange(offset_xy.detach(), "b v (h w) srf xy -> (b v) (srf xy) h w",srf=self.cfg.num_surfaces,xy=2, h=h, w=w), size=(64, 64), mode='bilinear', align_corners=True
                )
                offset_xy_cube = rearrange(offset_xy_cube, "(b v) (srf xy) h w -> b v (h w) srf xy", b=b, v=v, h=64, w=64,srf=self.cfg.num_surfaces,xy=2)  # b v 4096 1 2

                pixel_size = 1 / \
                    torch.tensor((64, 64), dtype=torch.float32, device=device)
                xy_ray_cube = xy_ray_cube + (offset_xy_cube - 0.5) * pixel_size # b v n srf 2
                sh_input_images_cube = F.interpolate(
                    rearrange(context["image"], "b v c h w -> (b v) c h w"),
                    size=(64, 64),
                    mode='bilinear',
                    align_corners=True,
                )
                sh_input_images_cube = rearrange(
                    sh_input_images_cube, "(b v) c h w -> b v c h w", b=b, v=v, h=64, w=64
                )

                gaussians_cube = self.gaussian_cube_adapter.forward(
                            rearrange(context["extrinsics"],
                                    "b v i j -> b v () () () i j"),
                            rearrange(context["intrinsics"],
                                    "b v i j -> b v () () () i j"),
                            rearrange(xy_ray_cube, "b v r srf xy -> b v r srf () xy"),
                            rearrange(depth_cube, "b v h w -> b v (h w) ()"), # b v 64 64
                            raw_opacities_cube_cp, # b v n 1
                            rearrange(
                                gaussians_cube_cp[..., 2:],
                                "b v r srf c -> b v r srf () c",
                            ),
                            (h, w),
                            input_images=sh_input_images_cube if self.cfg.init_sh_input_img else None,
                            mask = mask,
                            coef = coef
                        )

                # gaussian_cube = einsum("ijk, ijl, ijm -> ijklm", gaussians_z, gaussians_x, gaussians_y) # bv, c, 128, 64, 64
                # gaussian_cube = gaussian_cube*mask_dict["mask"].unsqueeze(1) # bv, c, 128, 64, 64
                # gaussian_cube = rearrange(gaussian_cube, "(bv) c d h w-> b v c d h w", b=b, v=v)

                # this should be added in ga_adapter after mapping to world coordinates
                # gaussian_cube = gaussian_cube.sum(dim=1) # sum over v, [B, C, D, H, W]


        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        gaussians = Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )

        gaussian_cube = Gaussians(
            rearrange(
                gaussians_cube.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians_cube.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians_cube.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians_cube.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )

        if self.cfg.return_depth:
            # return depth prediction for supervision
            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, V, H, W]

            return {
                "gaussians": gaussians,
                "depths": depths
            }
        if self.gs_cube:
            # return gaussians and gaussian_cube
            return gaussians_cube
        return gaussians

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            return batch

        return data_shim

    @property
    def sampler(self):
        return None

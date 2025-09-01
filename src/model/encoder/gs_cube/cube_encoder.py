
'''
Cube Encoder for GS-Cube
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from Swin3D.models import Swin3DUNet
from ....geometry.projection import get_world_rays
from ....geometry.vggt_geometry import unproject_depth_map_to_point_map
from ....geometry.projection import sample_image_grid
from einops import rearrange, repeat
import MinkowskiEngine as ME
from torch_scatter import scatter_max
from Swin3D.modules.mink_layers import assign_feats


class GSCubeInput():
    def __init__(self, feats, coords_centers, positions, imgs, cell_sizes, xyz_min, xyz_max, device='cpu'):
        self.sp = ME.SparseTensor(
            features=feats,
            coordinates=coords_centers,
            device=device
        )
        self.coords_sp = ME.SparseTensor(
            features=positions,
            coordinate_map_key=self.sp.coordinate_map_key, 
            coordinate_manager=self.sp.coordinate_manager,
            device=device
        )
        self.imgs = imgs  # BxVxCxHxW, range: [0, 1]
        self.cell_sizes = cell_sizes
        self.xyz_min = xyz_min
        self.xyz_max = xyz_max

    @property
    def coords_centers(self):
        return self.sp.C # batch_ind, xyz in grid centers

    @ property
    def position_centers(self):
        return torch.floor(self.coords_sp.F[:, 1:4])+0.5
    
    @property
    def true_positions(self):
        return self.coords_sp.F[:, :4] # batch_ind, xyz
     
    @property
    def rgbs(self):
        return self.coords_sp.F[:, 4:7]  # RGB values
    
    def retrieve_rgb_from_batch_coords(self, coords):
        '''
        Retrieve RGB features from coordinates that should be covered by sp.C.
        coords: Nx4, batch_ind, xyz
        '''
        # get the indices of coords in whole_coords
        rgbs = self.coords_sp.features_at_coordinates(coords.float())[:,4:7]
        return rgbs

    def retrieve_positions_from_batch_coords(self, coords):
        '''
        Retrieve positions from coordinates that should be covered by sp.C.
        coords: Nx4, batch_ind, xyz
        '''
        positions = (coords[:, 1:4] + 0.5)*self.cell_sizes  # convert to grid centers
        return positions

class GSCubeHead(nn.Module):
    def __init__(self, inchannels=48, num_gaussian_parameters=48, gpc=4):
        super(GSCubeHead, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(inchannels, num_gaussian_parameters),
            nn.GELU(),
            nn.Linear(num_gaussian_parameters, gpc*num_gaussian_parameters),
        )
        # self.l2 = nn.Conv1d(
        #     in_channels=num_gaussian_parameters, 
        #     out_channels=num_gaussian_parameters * gpc, 
        #     kernel_size=1, 
        #     bias=True,
        #     groups=num_gaussian_parameters
        # )
        
    def forward(self, x):
        # x = self.l1(x)
        # x = rearrange(x, "n c -> n c ()")
        # x = self.l2(x)
        # x = rearrange(x, "n c l -> n (c l)")
        x = self.l1(x)
        return x

class GSCubeEncoder(Swin3DUNet):
    def __init__(
            self, depths, channels, num_heads, window_sizes,
            quant_size, drop_path_rate=0.2, up_k=3,
            num_layers=5, num_classes=13, stem_transformer=True, 
            down_strides=[2,2], upsample='linear', knn_down=True,
            in_channels=6, cRSE='XYZ_RGB', fp16_mode=0, 
            num_gaussian_parameters=48, gpc = 4, cell_scale = 1):
        super(GSCubeEncoder, self).__init__(
            depths=depths, channels=channels, num_heads=num_heads, 
            window_sizes=window_sizes, quant_size=quant_size, 
            drop_path_rate=drop_path_rate, up_k=up_k, 
            num_layers=num_layers, num_classes=num_classes, 
            stem_transformer=stem_transformer, upsample=upsample, 
            first_down_stride=down_strides[0], 
            other_down_stride=down_strides[1], 
            knn_down=knn_down, 
            in_channels=in_channels, cRSE=cRSE, fp16_mode=fp16_mode
        )
        # self.gs_cube_head = nn.Sequential(
        #         nn.Linear(48, num_gaussian_parameters),
        #         nn.GELU(),
        #         nn.Linear(num_gaussian_parameters, num_gaussian_parameters)
        #     )
        self.gs_cube_head = GSCubeHead(inchannels=channels[0], num_gaussian_parameters=num_gaussian_parameters, gpc = gpc)
        self.gpc = gpc
        self.cell_scale = cell_scale

    def encode(self, sp, coords_sp): 
        # sp: MinkowskiEngine SparseTensor for feature input
        # sp.F: input features,         NxC
        # sp.C: input coordinates,      Nx4
        # coords_sp: MinkowskiEngine SparseTensor for position and feature embedding
        # coords_sp.F: embedding
        #       Batch: 0,...0,1,...1,2,...2,...,B,...B
        #       XYZ:   in Voxel Scale
        #       RGB:   in [-1,1]
        #       NORM:  in [-1,1]
        #       Batch, XYZ:             Nx4
        #       Batch, XYZ, RGB:        Nx7
        #       Batch, XYZ, RGB, NORM:  Nx10
        # coords_sp.C: input coordinates: Nx4
        sp_stack = []
        coords_sp_stack = []
        sp = self.stem_layer(sp)
        if self.layer_start > 0:
            sp_stack.append(sp)
            coords_sp_stack.append(coords_sp)
            sp, coords_sp = self.downsample(sp, coords_sp)

        for i, layer in enumerate(self.layers):
            coords_sp_stack.append(coords_sp)
            sp, sp_down, coords_sp = layer(sp, coords_sp)
            sp_stack.append(sp)
            assert (coords_sp.C == sp_down.C).all()
            sp = sp_down
        return sp_stack, coords_sp_stack, sp_down.C.shape[0]       

    def decode(self, sp_stack, coords_sp_stack):
        sp = sp_stack.pop()
        coords_sp = coords_sp_stack.pop()
        for i, upsample in enumerate(self.upsamples):
            sp_i = sp_stack.pop()
            coords_sp_i = coords_sp_stack.pop()
            sp = upsample(sp, coords_sp, sp_i, coords_sp_i)
            coords_sp = coords_sp_i

        # output = self.classifier(sp.F)
        return sp, coords_sp

    # def forward(self, sp, coords_sp):
        
    #     sp_stack, coords_sp_stack = self.encode(sp, coords_sp)
    #     sp_f, output = self.decode(sp_stack, coords_sp_stack)
        
    #     return sp_f, output
    def forward(self, imgs, 
                depth, feats, 
                extrinsics=None, 
                intrinsics=None,
                depth_min=0.5,
                depth_max=100.0,
                num_depth=128,
                return_perview=False,
                vggt_meta=False):
        '''
        imgs: context images, shape: BxVxCxHxW, range: [0, 1]
        depth: depth map, shape: BxVxHxW
        feats: additional features, shape: BxVxLxHxW
        '''
        # s1: get world coordinates from depth
        b,v,_,h,w = imgs.shape
        device = imgs.device
        if vggt_meta:
            _, coordinates = sample_image_grid((h, w), device)
            xyz_world = unproject_depth_map_to_point_map(
                coordinates = repeat(coordinates.float(), "h w c -> (b v) h w c", b=b, v=v),
                depth_map=rearrange(depth, "b v h w -> (b v) h w", h=h, w=w),
                extrinsics_cam=rearrange(extrinsics, "b v i j -> (b v) i j").inverse(),
                intrinsics_cam=rearrange(intrinsics, "b v i j -> (b v) i j")
            )
            xyz_world = rearrange(xyz_world, "(b v) h w c -> (b v) (h w) c", b=b, v=v, h=h, w=w, c=3)
        else:
            xy_ray, _ = sample_image_grid((h, w), device)
            xy_ray = torch.broadcast_to(xy_ray, (b, v, h, w, 2))
            origins, directions = get_world_rays(
                rearrange(xy_ray,"b v h w xy->(b v) (h w) xy"), 
                rearrange(extrinsics,"b v h w->(b v) () h w"), 
                rearrange(intrinsics,"b v h w->(b v) () h w"))
            depth = rearrange(depth, "b v h w->(b v) (h w)")
            xyz_world = origins + directions * depth[..., None] # bv, hw, 3
        # s2: voxelization, each cell contains at most one point
        gs_cube_input, gs_cube_input_perview = self.voxelize(xyz_world, 
                      rearrange(feats,"b v l h w -> b (v h w) l"), 
                      rearrange(imgs,"b v c h w -> b (v h w) c"), 
                      num_depth, b=b, v=v, h=h, w=w, return_perview=return_perview)
        # gs_cube_input, gs_cube_input_perview = self.extensible_voxelization(xyz_world, 
        #               rearrange(feats,"b v l h w -> b (v h w) l"), 
        #               rearrange(imgs,"b v c h w -> b (v h w) c"), 
        #               num_depth, b=b, v=v, h=h, w=w, return_perview=return_perview)
        # s3: encode
        sp_stack, coords_sp_stack, nog_min = self.encode(gs_cube_input.sp, gs_cube_input.coords_sp)
        # s4: decode
        sp, coords_sp = self.decode(sp_stack, coords_sp_stack)
        spf = self.gs_cube_head(sp.F)  # [N, KxC]
        sp = assign_feats(sp, spf)
        nog_pb = self.gpc*sp.C.shape[0]//b
        # nog_pv = self.gpc*sp.C.shape[0]//(b * v)
        nog_min = nog_min // b
        if return_perview:
            return sp, coords_sp, gs_cube_input, gs_cube_input_perview, nog_pb, nog_min
        return sp, gs_cube_input.coords_sp, gs_cube_input, None, nog_pb, nog_min

    def voxelize(self, xyz_world, feats, imgs, num_depth, b=None, v=None, h=None, w=None, return_perview=False):
        '''
        some notes about intermidiate variables:
        xyz_world: world coordinates, shape: (b, v*h*w, 3), range: physical distances
        xyz_nomralized: normalized coordinates, shape: (b, v*h*w, 3), range: [0, 1]
        xyz_scaled: true positions in a 3D mesh grid that is a float type, shape: (b, v*h*w, 3)
        selected_ind: indices of selected points.
        It is noted that if we voxelize the points with shape H,W,D, we have to use randomly sampling to ensure each voxel has at most one point.
        However, this is not an ideal case, since we may lose some points. We do this to make sure the voxelization is efficient.
        There are some other options such as Adaptive voxelization: we can choose a proper xyz_min, xyz_max, cell_size to include only one point in each voxel. This doesn't need the random sampling.
        Moreover, we can also impose hierarchical voxelization, which means we can first voxelize the points with a coarse grid, and then refine the grids, where each cell contains K>>1 points, to finer ones.
        In this case, we have to be careful about the parents and sons' coordinates, sons' coordinates cannot be float. There needs an extra sparse convolution, unaware of coordinate system, to aggregate these finer cells.
        '''
        
        device = xyz_world.device
        '''get grid_coords per batch'''
        xyz_world = rearrange(xyz_world, "(b v) (h w) xyz -> b (v h w) xyz", b=b, v=v, h=h, w=w)
        xyz_min = xyz_world.min(dim=1, keepdim=True)[0]
        xyz_max = xyz_world.max(dim=1, keepdim=True)[0]
        assert (xyz_max > xyz_min).all(), "xyz_max should be larger than xyz_min. now xyz_min: {}, xyz_max: {}".format(xyz_min, xyz_max)
        xyz_normalized = (xyz_world - xyz_min) / (xyz_max - xyz_min)
        # we should have hxwxnum_depth grid cells
        grid_shape = torch.tensor([h, w, num_depth], dtype=torch.float32, device=device)
        grid_shape = grid_shape*self.cell_scale
        cell_sizes = (xyz_max - xyz_min) / grid_shape
        cell_sizes = cell_sizes.squeeze(1) 
        # cell ind:0 -> h-1, 0->w-1, 0->num_depth-1
        xyz_scaled = xyz_normalized * (grid_shape-1e-4)
        grid_coords = torch.floor(xyz_scaled).long()
        # gradient_placeholder = xyz_scaled-xyz_scaled.detach()

        '''get grid_coords per view'''
        if return_perview:
            view_ind = torch.cat([torch.full((b, h*w, 1), view) for view in range(v)], dim=1).to(device) # b (v h w) 1
            grid_coords_view = torch.cat([grid_coords, view_ind], dim=-1)  # b (v h w) 4
            position_all_views = []
            coords_all_views = []
            feats_all_views = []

        '''get per batch and per view sparse tensors'''
        batch_positions =  []
        batch_coords_centers = []
        batch_feats = []

        
        for batch_idx in range(xyz_world.shape[0]):  
            '''randomly select one within each voxel'''
            # Ensure each cell has at most one point
            unique_coords, inverse_ind = torch.unique(
                grid_coords[batch_idx,:,:], 
                dim=0, 
                return_inverse=True, 
                return_counts=False, 
                sorted=False
            )
            # randomly select one point in each voxel by choosing the max random value
            rand_val = torch.rand(grid_coords[batch_idx,:,:].size(0), device=device)
            _, selected_ind = scatter_max(
                rand_val,
                inverse_ind,
                dim=0,
                dim_size=unique_coords.shape[0]
            )
            unique_grid_coords = grid_coords[batch_idx,selected_ind,:]
            feats_unique = feats[batch_idx, selected_ind, :]
            imgs_unique = imgs[batch_idx, selected_ind, :]
            positions = xyz_scaled[batch_idx, selected_ind, :]
            # concatenate the batch index with coords
            coords_centers = torch.cat([torch.full((unique_grid_coords.shape[0], 1), batch_idx).to(device), unique_grid_coords], dim=-1)  # Nx4
            positions = torch.cat([torch.full((positions.shape[0], 1), batch_idx).to(device), positions, imgs_unique], dim=-1)  # Nx7
            batch_coords_centers.append(coords_centers)
            batch_positions.append(positions)    
            batch_feats.append(feats_unique)

            if return_perview:
                ''' the same process for each view'''
                unique_coords_view, inverse_ind_view = torch.unique(
                    grid_coords_view[batch_idx,:,:], 
                    dim=0, 
                    return_inverse=True, 
                    return_counts=False, 
                    sorted=False
                )
                rand_val_view = torch.rand(grid_coords_view[batch_idx,:,:].size(0), device=device)
                _, selected_ind_view = scatter_max(
                    rand_val_view,
                    inverse_ind_view,
                    dim=0,
                    dim_size=unique_coords_view.shape[0]
                )
                unique_grid_coords_view = grid_coords_view[batch_idx,selected_ind_view,:]
                feats_unique_view = feats[batch_idx, selected_ind_view, :]
                imgs_unique_view = imgs[batch_idx, selected_ind_view, :]
                positions_view = xyz_scaled[batch_idx, selected_ind_view, :]
                # concatenate the batch index with coords
                coords_centers_view = torch.cat([torch.full((unique_grid_coords_view.shape[0], 1), batch_idx).to(device), unique_grid_coords_view], dim=-1)  # Nx4
                positions_view = torch.cat([torch.full((positions_view.shape[0], 1), batch_idx).to(device), positions_view, imgs_unique_view], dim=-1)
                coords_per_view = []
                position_per_view = []
                feats_per_view = []
                for view_id in range(v):
                    ind = torch.where(coords_centers_view[:,4] == view_id)[0]
                    coords_per_view.append(coords_centers_view[ind,:4])  # Nx4
                    position_per_view.append(positions_view[ind,:])  # Nx7
                    feats_per_view.append(feats_unique_view[ind,:])  # NxC
                coords_all_views.append(coords_per_view)
                position_all_views.append(position_per_view)
                feats_all_views.append(feats_per_view)
        # stack the coords
        batch_coords_centers = torch.cat(batch_coords_centers, dim=0)  # [N,4]
        batch_positions = torch.cat(batch_positions, dim=0)  # [N, 4]     
        batch_feats = torch.cat(batch_feats, dim=0)  # [N, C]
        
        gs_cube_tensor = GSCubeInput(
            feats=batch_feats, 
            coords_centers=batch_coords_centers, 
            positions=batch_positions, 
            imgs=None,
            cell_sizes=cell_sizes,
            xyz_min=xyz_min,
            xyz_max=xyz_max,
            device=device
        )
        if return_perview:
            gs_cube_tensor_bv = {}
            for batch_idx in range(b):
                gs_cube_tensor_bv[batch_idx] = {}
                for view_id in range(v):
                    gs_cube_tensor_bv[batch_idx][view_id] = GSCubeInput(
                        feats=feats_all_views[batch_idx][view_id], 
                        coords_centers=coords_all_views[batch_idx][view_id], 
                        positions=position_all_views[batch_idx][view_id], 
                        imgs=None,
                        cell_sizes=cell_sizes,
                        xyz_min=xyz_min[batch_idx:batch_idx+1],
                        xyz_max=xyz_max[batch_idx:batch_idx+1],
                        device=device
                    )
            return gs_cube_tensor, gs_cube_tensor_bv
        return gs_cube_tensor, None
    def extensible_voxelization(self, xyz_world, feats, imgs, num_depth, b=None, v=None, h=None, w=None, return_perview=False):
        '''
        compared with plain voxelization, this function can handle unknown input number of views
        ''' 
        
        if v>=2:
            device = xyz_world.device 
            grid_shape = torch.tensor([h, w, num_depth], dtype=torch.float32, device=device)
            grid_shape = grid_shape*self.cell_scale
            imgs_reshape = rearrange(imgs, "b (v h w) c -> b v (h w) c", b=b, v=v, h=h, w=w)
            xyz_world_reshape = rearrange(xyz_world, "(b v) (h w) xyz -> b v (h w) xyz", b=b, v=v, h=h, w=w)
            feats_reshape = rearrange(feats, "b (v h w) l -> b v (h w) l", b=b, v=v, h=h, w=w)

            xyz_world_anchor = xyz_world_reshape[:, :2, :, :]  # use the first 2 views as the anchor
            imgs_anchor = imgs_reshape[:, :2, :, :]  # use the first 2 views as the anchor
            feats_anchor  = feats_reshape[:, :2, :, :]
            xyz_world_anchor = rearrange(xyz_world_anchor, "b v (h w) xyz -> b (v h w) xyz", b=b, v=2, h=h, w=w)
            imgs_anchor = rearrange(imgs_anchor, "b v (h w) c -> b (v h w) c", b=b, v=2, h=h, w=w)
            feats_anchor = rearrange(feats_anchor, "b v (h w) l -> b (v h w) l", b=b, v=2, h=h, w=w)

            
            xyz_world_rest = xyz_world_reshape[:, 2:, :, :]  # the rest views
            imgs_rest = imgs_reshape[:, 2:, :, :]
            feats_rest = feats_reshape[:, 2:, :, :]
            xyz_world_rest = rearrange(xyz_world_rest, "b v (h w) xyz -> b (v h w) xyz", b=b, v=v-2, h=h, w=w)
            imgs_rest = rearrange(imgs_rest, "b v (h w) c -> b (v h w) c", b=b, v=v-2, h=h, w=w)
            feats_rest = rearrange(feats_rest, "b v (h w) l -> b (v h w) l", b=b, v=v-2, h=h, w=w)

            '''get grid_coords per batch'''
           
            xyz_min = xyz_world_anchor.min(dim=1, keepdim=True)[0]
            xyz_max = xyz_world_anchor.max(dim=1, keepdim=True)[0]
            cell_sizes = (xyz_max - xyz_min) / grid_shape
            cell_sizes = cell_sizes.squeeze(1) 
            assert (xyz_max > xyz_min).all(), "xyz_max should be larger than xyz_min"
    
            xyz_world_rest_normalized = (xyz_world_rest - xyz_min) / (xyz_max - xyz_min)
            xyz_world_anchor_normalized = (xyz_world_anchor - xyz_min) / (xyz_max - xyz_min)
            # remove overlapping points
            xyz_world_rest_scaled = xyz_world_rest_normalized * (grid_shape-1e-4)
            xyz_world_anchor_scaled = xyz_world_anchor_normalized * (grid_shape-1e-4)
            xyz_world_rest_coords = torch.floor(xyz_world_rest_scaled).long()
            xyz_world_anchor_coords = torch.floor(xyz_world_anchor_scaled).long()

            xyz_non_overlap = []
            xyz_non_overlap_scaled = []
            imgs_non_overlap = []
            feats_non_overlap = []
            
            for bi in range(xyz_world_rest_coords.shape[0]):
                xyz_b = []
                xyz_scaled_b = []
                imgs_b = []
                feats_b = []
                for ni in range(xyz_world_rest_coords.shape[1]):
                    match = torch.where((xyz_world_anchor_coords[bi]-xyz_world_rest_coords[bi, ni]).sum(-1) == 0)[0]
                    if not match.any():
                        xyz_b.append(xyz_world_rest_coords[bi, ni])
                        xyz_scaled_b.append(xyz_world_rest_scaled[bi, ni])
                        imgs_b.append(imgs_rest[bi, ni])
                        feats_b.append(feats_rest[bi, ni])
                if len(xyz_b) > 0:
                    xyz_non_overlap.append(torch.stack(xyz_b, dim=0))
                    xyz_non_overlap_scaled.append(torch.stack(xyz_scaled_b, dim=0))
                    imgs_non_overlap.append(torch.stack(imgs_b, dim=0))
                    feats_non_overlap.append(torch.stack(feats_b, dim=0))
                    print("batch {} has {} non-overlapping points".format(bi, len(xyz_b)))
            if len(xyz_non_overlap) > 0:
                xyz_world_rest_coords = torch.stack(xyz_non_overlap, dim=0)
                xyz_world_rest_scaled = torch.stack(xyz_non_overlap_scaled, dim=0)
                imgs_rest = torch.stack(imgs_non_overlap, dim=0)
                feats_rest = torch.stack(feats_non_overlap, dim=0)
                print(xyz_world_rest_coords.shape[1], "non-overlapping points in the rest views")
            
            
            # cell ind:0 -> h-1, 0->w-1, 0->num_depth-1
            xyz_scaled = torch.cat([xyz_world_anchor_scaled, xyz_world_rest_scaled], dim=1)  # b (v h w) xyz
            grid_coords = torch.cat([xyz_world_anchor_coords, xyz_world_rest_coords], dim=1)  # b (v h w) xyz
            imgs = torch.cat([imgs_anchor, imgs_rest], dim=1)  # b (v h w) c
            feats = torch.cat([feats_anchor, feats_rest], dim=1)  #
            # xyz_scaled = xyz_world_anchor_scaled
            # grid_coords = xyz_world_anchor_coords
            # imgs = imgs_anchor
            # feats = feats_anchor
            '''get grid_coords per view'''
            if return_perview:
                view_ind = torch.cat([torch.full((b, h*w, 1), view) for view in range(v)], dim=1).to(device) # b (v h w) 1
                grid_coords_view = torch.cat([grid_coords, view_ind], dim=-1)  # b (v h w) 4
                position_all_views = []
                coords_all_views = []
                feats_all_views = []

            '''get per batch and per view sparse tensors'''
            batch_positions =  []
            batch_coords_centers = []
            batch_feats = []
            
            for batch_idx in range(xyz_world_anchor.shape[0]):  
                '''randomly select one within each voxel'''
                # Ensure each cell has at most one point
                unique_coords, inverse_ind = torch.unique(
                    grid_coords[batch_idx,:,:], 
                    dim=0, 
                    return_inverse=True, 
                    return_counts=False, 
                    sorted=False
                )
                # randomly select one point in each voxel by choosing the max random value
                rand_val = torch.rand(grid_coords[batch_idx,:,:].size(0), device=device)
                _, selected_ind = scatter_max(
                    rand_val,
                    inverse_ind,
                    dim=0,
                    dim_size=unique_coords.shape[0]
                )
                unique_grid_coords = grid_coords[batch_idx,selected_ind,:]
                feats_unique = feats[batch_idx, selected_ind, :]
                imgs_unique = imgs[batch_idx, selected_ind, :]
                positions = xyz_scaled[batch_idx, selected_ind, :]
                # concatenate the batch index with coords
                coords_centers = torch.cat([torch.full((unique_grid_coords.shape[0], 1), batch_idx).to(device), unique_grid_coords], dim=-1)  # Nx4
                positions = torch.cat([torch.full((positions.shape[0], 1), batch_idx).to(device), positions, imgs_unique], dim=-1)  # Nx7
                batch_coords_centers.append(coords_centers)
                batch_positions.append(positions)    
                batch_feats.append(feats_unique)

                if return_perview:
                    ''' the same process for each view'''
                    unique_coords_view, inverse_ind_view = torch.unique(
                        grid_coords_view[batch_idx,:,:], 
                        dim=0, 
                        return_inverse=True, 
                        return_counts=False, 
                        sorted=False
                    )
                    rand_val_view = torch.rand(grid_coords_view[batch_idx,:,:].size(0), device=device)
                    _, selected_ind_view = scatter_max(
                        rand_val_view,
                        inverse_ind_view,
                        dim=0,
                        dim_size=unique_coords_view.shape[0]
                    )
                    unique_grid_coords_view = grid_coords_view[batch_idx,selected_ind_view,:]
                    feats_unique_view = feats[batch_idx, selected_ind_view, :]
                    imgs_unique_view = imgs[batch_idx, selected_ind_view, :]
                    positions_view = xyz_scaled[batch_idx, selected_ind_view, :]
                    # concatenate the batch index with coords
                    coords_centers_view = torch.cat([torch.full((unique_grid_coords_view.shape[0], 1), batch_idx).to(device), unique_grid_coords_view], dim=-1)  # Nx4
                    positions_view = torch.cat([torch.full((positions_view.shape[0], 1), batch_idx).to(device), positions_view, imgs_unique_view], dim=-1)
                    coords_per_view = []
                    position_per_view = []
                    feats_per_view = []
                    for view_id in range(v):
                        ind = torch.where(coords_centers_view[:,4] == view_id)[0]
                        coords_per_view.append(coords_centers_view[ind,:4])  # Nx4
                        position_per_view.append(positions_view[ind,:])  # Nx7
                        feats_per_view.append(feats_unique_view[ind,:])  # NxC
                    coords_all_views.append(coords_per_view)
                    position_all_views.append(position_per_view)
                    feats_all_views.append(feats_per_view)
            # stack the coords
            batch_coords_centers = torch.cat(batch_coords_centers, dim=0)  # [N,4]
            batch_positions = torch.cat(batch_positions, dim=0)  # [N, 4]     
            batch_feats = torch.cat(batch_feats, dim=0)  # [N, C]
            
            gs_cube_tensor = GSCubeInput(
                feats=batch_feats, 
                coords_centers=batch_coords_centers, 
                positions=batch_positions, 
                imgs=None,
                cell_sizes=cell_sizes,
                xyz_min=xyz_min,
                xyz_max=xyz_max,
                device=device
            )
            if return_perview:
                gs_cube_tensor_bv = {}
                for batch_idx in range(b):
                    gs_cube_tensor_bv[batch_idx] = {}
                    for view_id in range(v):
                        gs_cube_tensor_bv[batch_idx][view_id] = GSCubeInput(
                            feats=feats_all_views[batch_idx][view_id], 
                            coords_centers=coords_all_views[batch_idx][view_id], 
                            positions=position_all_views[batch_idx][view_id], 
                            imgs=None,
                            cell_sizes=cell_sizes,
                            xyz_min=xyz_min[batch_idx:batch_idx+1],
                            xyz_max=xyz_max[batch_idx:batch_idx+1],
                            device=device
                        )
                return gs_cube_tensor, gs_cube_tensor_bv
            return gs_cube_tensor, None

        else:
            # if v <= 2, we can use the plain voxelization
            return self.voxelize(xyz_world, feats, imgs, num_depth, b=b, v=v, h=h, w=w, return_perview=return_perview)

if __name__ == "__main__":
    import MinkowskiEngine as ME
    import numpy as np
    from tqdm import tqdm
    import os
    np.random.seed(42)
    torch.manual_seed(42)
    os.environ['CUDA_LAUNCH_BLOCKING']="1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"

    # Example usage
    encoder = GSCubeEncoder(
        depths=[2, 2, 2, 2, 2],
        channels=[16, 16, 32, 64, 64],
        num_heads=[2, 2, 4, 8, 8],
        window_sizes=[5, 7, 7, 7, 7],
        quant_size=4,
        in_channels = 9,
        first_down_stride=3,
        knn_down=True,
        upsample= 'linear_attn',
        cRSE='XYZ_RGB',
        up_k= 3,
        num_classes=13,
        stem_transformer=True,
        fp16_mode=0,
    )
    encoder.cuda()

    batch_size = 2
    num_points = 256*256  # Non-zero points per batch
    in_channels = 9    # Default for sp features
    embedding_dims = 6 # e.g., XYZ(3) + RGB(3) for coords_sp

    # 1. Generate batch indices and random coordinates
    xyz_coords = torch.randint(0, 128, (2*batch_size * num_points, 3))  # Random voxel positions (0-99)
    xyz_coords = torch.unique(xyz_coords, dim=0)  # Ensure unique coordinates
    xyz_coords = xyz_coords[:batch_size * num_points]  # Limit to batch size
    batch_indices = torch.cat([torch.full((num_points, 1), i) for i in range(batch_size)])
    
    coords = torch.cat([batch_indices, xyz_coords], dim=1)  # Shape: [N, 4]

    # 2. Generate features with torch.randn
    sp_features = 0.5*torch.randn(batch_size * num_points, in_channels)  # For sp.F
    coords_sp_features = torch.cat([coords,-0.7*torch.ones(batch_size * num_points, 3)], dim=-1)  # For coords_sp.F
    # 3. Convert to Minkowski SparseTensors
    sp = ME.SparseTensor(
        features=sp_features,
        coordinates=coords,
        device="cuda"  # Optional GPU
    )

    coords_sp = ME.SparseTensor(
        features=coords_sp_features,
        coordinate_map_key=sp.coordinate_map_key, 
        coordinate_manager=sp.coordinate_manager,
        device="cuda"
    )

    # Run forward pass
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    encoder.train()
    for _ in tqdm(range(1000)):
        optimizer.zero_grad()
        sp_f, output = encoder(sp, coords_sp)
        # print('forward output shape:', output.shape)
        loss = nn.MSELoss()(output, torch.randn_like(output).cuda())  # Dummy loss
        loss.backward()
        if _ % 10 == 0:
            print(f"Step {_}, Loss: {loss.item()}")
    
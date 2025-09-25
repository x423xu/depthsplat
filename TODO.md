1. Depth distillation from vggt meta.
Before distilling the depth, we need to make sure whether the depthsplat generate consistent depth when input with mutiple images.
emmm, how to evaluate the inconsistency? the only way is to evaluate on NYUv2 dataset. Compare the depth consistency from two methods: depthsplat and vggt depth.

- [ ] dual head
- [ ] comparision: depth22+scale4 vs 
- [ ] random scale

Nan values is caused by the depth Nan values. Reasons might be: 1. Checkpoitn in attention module influences the grads. Turn it off. 2. The learning rate for updating depth model is too large, leading to grad explosion. A lr_depth=1e-7 works fine till now.

the answer is depth estimation moves, making the swin3d has zero entries in certain windows, this is also related to knn or woknn, causing the division by zero.

Why previous woknn works but now it fails even making all settings the same. Maybe cell_scale=8 -> checkpoint=True, cell_scale=4 -> checkpoint=False
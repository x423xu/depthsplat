1. Depth distillation from vggt meta.
Before distilling the depth, we need to make sure whether the depthsplat generate consistent depth when input with mutiple images.
emmm, how to evaluate the inconsistency? the only way is to evaluate on NYUv2 dataset. Compare the depth consistency from two methods: depthsplat and vggt depth.

2. 
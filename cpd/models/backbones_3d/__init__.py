from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import VoxelBackBone8x,VoxelResBackBone8x
__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    'VoxelResBackBone8x': VoxelResBackBone8x,
}

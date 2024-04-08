from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import VoxelBackBone8x,SimpleVoxelBackBone8x,VoxelResBackBone8x, VirConv8x, VirConvRes8x, VirConv2x,TeVoxelBackBone8x,VirConv8xO, Voxel_MAE
from .votr_backbone import VoxelTransformer, VoxelTransformerV2, VoxelTransformerV3
__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'SimpleVoxelBackBone8x': SimpleVoxelBackBone8x,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    'VoxelResBackBone8x': VoxelResBackBone8x,
    'VoxelTransformer': VoxelTransformer,
    'VoxelTransformerV3': VoxelTransformerV3,
    'VirConv8x': VirConv8x,
    'VirConv8xO': VirConv8xO,
    'VirConvRes8x': VirConvRes8x,
    'VirConv2x': VirConv2x,
    'TeVoxelBackBone8x': TeVoxelBackBone8x,
    'Voxel_MAE': Voxel_MAE
}

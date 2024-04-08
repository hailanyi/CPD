from .roi_head_template import RoIHeadTemplate
from .voxel_rcnn_head import VoxelRCNNHead, VoxelRCNNProtoHead

__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'VoxelRCNNHead': VoxelRCNNHead,
    'VoxelRCNNProtoHead': VoxelRCNNProtoHead
}

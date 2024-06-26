import torch
import torch.nn as nn
from .roi_head_template import RoIHeadTemplate
from ...utils import common_utils, spconv_utils
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from torch.autograd import Variable
import torch.nn.functional as F
import copy
import numpy as np
from ...utils import box_coder_utils, common_utils, loss_utils
from ..model_utils.model_nms_utils import class_agnostic_nms
from .target_assigner.proposal_target_layer import ProposalTargetLayer
from ...utils.odiou_loss import odiou_3D
from ...utils.bbloss import bb_loss

class VoxelRCNNProtoHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, point_cloud_range=None, voxel_size=None, num_frames=1, num_class=1,
                 **kwargs):
        super().__init__(num_class=num_class, num_frames=num_frames, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        self.pool_cfg_mm = model_cfg.ROI_GRID_POOL_PROTO
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        LAYER_cfg_mm = self.pool_cfg_mm.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = LAYER_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [input_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name].QUERY_RANGES,
                nsamples=LAYER_cfg[src_name].NSAMPLE,
                radii=LAYER_cfg[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=LAYER_cfg[src_name].POOL_METHOD,
            )

            self.roi_grid_pool_layers.append(pool_layer)

            c_out += sum([x[-1] for x in mlps])

        c_out_mm = 0
        self.roi_grid_pool_layers_mm = nn.ModuleList()
        feat = self.pool_cfg_mm.get('FEAT_NUM', 1)
        for src_name in self.pool_cfg_mm.FEATURES_SOURCE:
            mlps = LAYER_cfg_mm[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [input_channels[src_name] * feat] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg_mm[src_name].QUERY_RANGES,
                nsamples=LAYER_cfg_mm[src_name].NSAMPLE,
                radii=LAYER_cfg_mm[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=LAYER_cfg_mm[src_name].POOL_METHOD,
            )

            self.roi_grid_pool_layers_mm.append(pool_layer)

            c_out_mm += sum([x[-1] for x in mlps])

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.shared_fc_layers = nn.Sequential(*shared_fc_list)

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL_PROTO.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out_mm
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.shared_fc_layers_mm = nn.Sequential(*shared_fc_list)

        self.shared_channel = pre_channel

        pre_channel = self.model_cfg.SHARED_FC[-1]
        cls_fc_list = []
        for k in range(0, self.model_cfg.CLS_FC.__len__()):
            cls_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.CLS_FC[k]

            if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        cls_fc_list.append(nn.Linear(pre_channel, self.num_class, bias=True))
        cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_layers = cls_fc_layers

        pre_channel = self.model_cfg.SHARED_FC[-1]
        reg_fc_list = []
        for k in range(0, self.model_cfg.REG_FC.__len__()):
            reg_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.REG_FC[k]

            if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        reg_fc_list.append(nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True))
        reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_layers = reg_fc_layers

        pre_channel = self.model_cfg.SHARED_FC[-1]
        cls_fc_list = []
        for k in range(0, self.model_cfg.CLS_FC.__len__()):
            cls_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.CLS_FC[k]

            if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        cls_fc_list.append(nn.Linear(pre_channel, self.num_class, bias=True))
        cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_layers_P = cls_fc_layers

        pre_channel = self.model_cfg.SHARED_FC[-1]
        reg_fc_list = []
        for k in range(0, self.model_cfg.REG_FC.__len__()):
            reg_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.REG_FC[k]

            if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        reg_fc_list.append(nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True))
        reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_layers_P = reg_fc_layers

        self.init_weights()
        self.ious = {0: [], 1: [], 2: [], 3: []}

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for stage_module in [self.cls_layers, self.reg_layers]:
            for m in stage_module.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        for stage_module in [self.cls_layers, self.reg_layers]:
            nn.init.normal_(stage_module[-1].weight, 0, 0.01)
            nn.init.constant_(stage_module[-1].bias, 0)
        for m in self.shared_fc_layers.modules():
            if isinstance(m, nn.Linear):
                init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """

        rois = batch_dict['rois'].clone()

        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)

        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            if src_name in ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']:

                cur_stride = batch_dict['multi_scale_3d_strides'][src_name]

                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

                # compute voxel center xyz and batch_cnt
                cur_coords = cur_sp_tensors.indices
                cur_voxel_xyz = common_utils.get_voxel_centers(
                    cur_coords[:, 1:4],
                    downsample_times=cur_stride,
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.point_cloud_range
                )  #
                cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
                for bs_idx in range(batch_size):
                    cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
                # get voxel2point tensor

                v2p_ind_tensor = spconv_utils.generate_voxel2pinds(cur_sp_tensors)

                # compute the grid coordinates in this scale, in [batch_idx, x y z] order
                cur_roi_grid_coords = roi_grid_coords // cur_stride
                cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
                cur_roi_grid_coords = cur_roi_grid_coords.int()
                # voxel neighbor aggregation
                pooled_features = pool_layer(
                    xyz=cur_voxel_xyz.contiguous(),
                    xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                    new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                    new_xyz_batch_cnt=roi_grid_batch_cnt,
                    new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                    features=cur_sp_tensors.features.contiguous(),
                    voxel2point_indices=v2p_ind_tensor
                )

                pooled_features = pooled_features.view(
                    -1, self.pool_cfg.GRID_SIZE ** 3,
                    pooled_features.shape[-1]
                )  # (BxN, 6x6x6, C)
                pooled_features_list.append(pooled_features)

        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)

        return ms_pooled_features

    def roi_grid_pool_mm(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """

        rois = batch_dict['rois'].clone()
        #rois[:, 3:5] = rois[:, 3:5]*0.5

        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)

        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg_mm.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg_mm.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers_mm[k]
            if src_name in ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']:

                cur_stride = batch_dict['multi_scale_3d_strides'][src_name]

                cur_sp_tensors = batch_dict['multi_scale_3d_features_mm'][src_name]

                # compute voxel center xyz and batch_cnt
                cur_coords = cur_sp_tensors.indices
                cur_voxel_xyz = common_utils.get_voxel_centers(
                    cur_coords[:, 1:4],
                    downsample_times=cur_stride,
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.point_cloud_range
                )  #
                cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
                for bs_idx in range(batch_size):
                    cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
                # get voxel2point tensor

                v2p_ind_tensor = spconv_utils.generate_voxel2pinds(cur_sp_tensors)

                # compute the grid coordinates in this scale, in [batch_idx, x y z] order
                cur_roi_grid_coords = roi_grid_coords // cur_stride
                cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
                cur_roi_grid_coords = cur_roi_grid_coords.int()
                # voxel neighbor aggregation
                pooled_features = pool_layer(
                    xyz=cur_voxel_xyz.contiguous(),
                    xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                    new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                    new_xyz_batch_cnt=roi_grid_batch_cnt,
                    new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                    features=cur_sp_tensors.features.contiguous(),
                    voxel2point_indices=v2p_ind_tensor
                )

                pooled_features = pooled_features.view(
                    -1, self.pool_cfg_mm.GRID_SIZE ** 3,
                    pooled_features.shape[-1]
                )  # (BxN, 6x6x6, C)
                pooled_features_list.append(pooled_features)

        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)

        return ms_pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def proto_loss(self, targets_dict0, targets_dict1):

        reg_valid_mask = targets_dict0['reg_valid_mask'].view(-1)
        code_size = self.box_coder.code_size
        shape = targets_dict0['gt_of_rois'].shape
        gt_boxes3d_ct = targets_dict0['gt_of_rois'].clone().view(shape[0] * shape[1], -1)[:, 0:7]
        rcnn_reg = targets_dict0['rcnn_reg']  # (rcnn_batch_size, C)
        rois = targets_dict0['rois'].clone().view(-1, code_size)[:, 0:7]
        rois[:, 0:3] = 0
        rois[:, 6] = 0
        fg_mask = (reg_valid_mask > 0)
        batch_box_preds0 = self.box_coder.decode_torch(rcnn_reg, rois).view(-1, code_size)

        css_score_mask = targets_dict0['additional_data']['css_score'].view(-1)
        css_score_mask_new = css_score_mask[fg_mask]

        if len(gt_boxes3d_ct[fg_mask]) == 0:
            b_loss0 = 0
        else:
            b_loss0 = bb_loss(batch_box_preds0[fg_mask], gt_boxes3d_ct[fg_mask])
            b_loss0 = b_loss0*css_score_mask_new
            b_loss0 = b_loss0.sum() / (fg_mask.sum() + 1)


        rcnn_reg1 = targets_dict1['rcnn_reg']  # (rcnn_batch_size, C)
        rois1 = targets_dict1['rois'].clone().view(-1, code_size)[:, 0:7]
        rois1[:, 0:3] = 0
        rois1[:, 6] = 0

        batch_box_preds1 = self.box_coder.decode_torch(rcnn_reg1, rois1).view(-1, code_size)
        batch_box_preds1 = batch_box_preds1.clone().detach()


        if len(gt_boxes3d_ct[fg_mask]) == 0:
            b_loss1 = 0
        else:
            b_loss1 = bb_loss(batch_box_preds0[fg_mask], batch_box_preds1[fg_mask])
            b_loss1 = b_loss1 * css_score_mask_new
            b_loss1 = b_loss1.sum() / (fg_mask.sum() + 1)


        begin_weight = 0.00001
        max_iter = 5000
        end_weight = 0.2

        if self.iter > max_iter:
            self.iter = max_iter
        this_weight = (self.iter / max_iter) * (end_weight - begin_weight) + begin_weight
        self.iter += 1

        b_loss1*=this_weight

        label = targets_dict0['rcnn_cls_labels'].view(-1)
        reg0 = targets_dict0['shared_features']
        reg1_in = targets_dict1['shared_features']
        reg1 = reg1_in.clone().detach()

        l = -torch.cosine_similarity(reg0, reg1, dim=-1)

        cls_valid_mask = (label >= 0).float()

        cls_valid_mask = cls_valid_mask*css_score_mask

        loss_sum = (l * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

        loss_sum = loss_sum*this_weight
        b_loss1 = b_loss1*this_weight

        final = b_loss0+b_loss1+loss_sum


        return final

    def get_box_reg_layer_loss(self, forward_ret_dict, ind=''):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask' + ind].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois' + ind][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src' + ind][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg' + ind]  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois' + ind]
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        outline_reg_mask = forward_ret_dict['additional_data']['css_score'].view(-1)
        reg_valid_mask2 = reg_valid_mask*outline_reg_mask

        fg_mask = (reg_valid_mask2 > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 7]
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(
                fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg' + ind] = rcnn_loss_reg.item()

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                # TODO: NEED to BE CHECK
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size)

                rcnn_boxes3d = common_utils.rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                ).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz

                loss_corner = loss_utils.get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7]
                )
                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner' + ind] = loss_corner.item()
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict, ind=''):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls' + ind]
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels' + ind].view(-1)

        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)

            css_score_mask = forward_ret_dict['additional_data']['css_score'].view(-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            cls_valid_mask2 = cls_valid_mask*css_score_mask

            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(),
                                                    reduction='none')

            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask2).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls' + ind: rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        loss_cfgs = self.model_cfg.LOSS_CONFIG

        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0

        rcnn_loss_cls0, cls_tb_dict0 = self.get_box_cls_layer_loss(self.forward_ret_dict['targets_dict0'])
        rcnn_loss += rcnn_loss_cls0
        rcnn_loss_reg0, reg_tb_dict0 = self.get_box_reg_layer_loss(self.forward_ret_dict['targets_dict0'])
        rcnn_loss += rcnn_loss_reg0

        proto_l=0
        rcnn_loss_cls1, cls_tb_dict1 = self.get_box_cls_layer_loss(self.forward_ret_dict['targets_dict1'])
        proto_l += 0.5*rcnn_loss_cls1
        rcnn_loss_reg1, reg_tb_dict1 = self.get_box_reg_layer_loss(self.forward_ret_dict['targets_dict1'])
        proto_l += 0.5*rcnn_loss_reg1

        proto_l += self.proto_loss(self.forward_ret_dict['targets_dict0'], self.forward_ret_dict['targets_dict1'])

        rcnn_loss+=(proto_l*loss_cfgs.LOSS_WEIGHTS['rcnn_proto_weight'])

        tb_dict['rcnn_loss'] = rcnn_loss.item()

        return rcnn_loss, tb_dict

    def forward(self, batch_dict):

        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            new_targets_dict = {}
            for item in targets_dict:
                if item == 'additional_data':
                    additional_data = {}
                    for this_k in targets_dict[item]:
                        additional_data[this_k] = targets_dict[item][this_k].clone()
                    new_targets_dict[item] = additional_data
                else:
                    new_targets_dict[item] = targets_dict[item].clone()

            proto_targets_dict = {}
            for item in targets_dict:
                if item == 'additional_data':
                    additional_data = {}
                    for this_k in targets_dict[item]:
                        additional_data[this_k] = targets_dict[item][this_k].clone()
                    proto_targets_dict[item] = additional_data
                else:
                    proto_targets_dict[item] = targets_dict[item].clone()

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        if self.training:
            # RoI aware pooling
            pooled_features_proto = self.roi_grid_pool_mm(batch_dict)  # (BxN, 6x6x6, C)

        # Box Refinement
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        shared_features = self.shared_fc_layers(pooled_features)
        rcnn_cls = self.cls_layers(shared_features)
        rcnn_reg = self.reg_layers(shared_features)


        if not self.training:

            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )

            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['batch_cls_preds'] = batch_cls_preds

            batch_dict['cls_preds_normalized'] = False
        else:

            new_targets_dict['rcnn_cls'] = rcnn_cls
            new_targets_dict['rcnn_reg'] = rcnn_reg
            new_targets_dict['shared_features'] = shared_features

            self.forward_ret_dict['targets_dict0'] = new_targets_dict


        if self.training:

            # Box Refinement
            pooled_features_proto = pooled_features_proto.view(pooled_features_proto.size(0), -1)
            shared_features_proto = self.shared_fc_layers_mm(pooled_features_proto)
            rcnn_cls_proto = self.cls_layers_P(shared_features_proto)
            rcnn_reg_proto = self.reg_layers_P(shared_features_proto)

            proto_targets_dict['rcnn_cls'] = rcnn_cls_proto
            proto_targets_dict['rcnn_reg'] = rcnn_reg_proto
            proto_targets_dict['shared_features'] = shared_features_proto

            self.forward_ret_dict['targets_dict1'] = proto_targets_dict

        return batch_dict

class VoxelRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, point_cloud_range=None, voxel_size=None, num_frames=1, num_class=1,
                 **kwargs):
        super().__init__(num_class=num_class, num_frames=num_frames, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = LAYER_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [input_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name].QUERY_RANGES,
                nsamples=LAYER_cfg[src_name].NSAMPLE,
                radii=LAYER_cfg[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=LAYER_cfg[src_name].POOL_METHOD,
            )

            self.roi_grid_pool_layers.append(pool_layer)

            c_out += sum([x[-1] for x in mlps])

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.shared_fc_layers = nn.Sequential(*shared_fc_list)


        self.shared_channel = pre_channel

        pre_channel = self.model_cfg.SHARED_FC[-1]
        cls_fc_list = []
        for k in range(0, self.model_cfg.CLS_FC.__len__()):
            cls_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.CLS_FC[k]

            if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        cls_fc_list.append(nn.Linear(pre_channel, self.num_class, bias=True))
        cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_layers = cls_fc_layers

        pre_channel = self.model_cfg.SHARED_FC[-1]
        reg_fc_list = []
        for k in range(0, self.model_cfg.REG_FC.__len__()):
            reg_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.REG_FC[k]

            if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        reg_fc_list.append(nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True))
        reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_layers = reg_fc_layers

        self.init_weights()
        self.ious = {0: [], 1: [], 2: [], 3: []}

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for stage_module in [self.cls_layers, self.reg_layers]:
            for m in stage_module.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        for stage_module in [self.cls_layers, self.reg_layers]:
            nn.init.normal_(stage_module[-1].weight, 0, 0.01)
            nn.init.constant_(stage_module[-1].bias, 0)
        for m in self.shared_fc_layers.modules():
            if isinstance(m, nn.Linear):
                init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """

        rois = batch_dict['rois'].clone()

        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)

        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            if src_name in ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']:

                cur_stride = batch_dict['multi_scale_3d_strides'][src_name]

                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

                # compute voxel center xyz and batch_cnt
                cur_coords = cur_sp_tensors.indices
                cur_voxel_xyz = common_utils.get_voxel_centers(
                    cur_coords[:, 1:4],
                    downsample_times=cur_stride,
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.point_cloud_range
                )  #
                cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
                for bs_idx in range(batch_size):
                    cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
                # get voxel2point tensor

                v2p_ind_tensor = spconv_utils.generate_voxel2pinds(cur_sp_tensors)

                # compute the grid coordinates in this scale, in [batch_idx, x y z] order
                cur_roi_grid_coords = roi_grid_coords // cur_stride
                cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
                cur_roi_grid_coords = cur_roi_grid_coords.int()
                # voxel neighbor aggregation
                pooled_features = pool_layer(
                    xyz=cur_voxel_xyz.contiguous(),
                    xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                    new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                    new_xyz_batch_cnt=roi_grid_batch_cnt,
                    new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                    features=cur_sp_tensors.features.contiguous(),
                    voxel2point_indices=v2p_ind_tensor
                )

                pooled_features = pooled_features.view(
                    -1, self.pool_cfg.GRID_SIZE ** 3,
                    pooled_features.shape[-1]
                )  # (BxN, 6x6x6, C)
                pooled_features_list.append(pooled_features)

        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)

        return ms_pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points
    def forward(self, batch_dict):

        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        # Box Refinement
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        shared_features = self.shared_fc_layers(pooled_features)
        rcnn_cls = self.cls_layers(shared_features)
        rcnn_reg = self.reg_layers(shared_features)

        if not self.training:

            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )

            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['cls_preds_normalized'] = False
        else:

            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict

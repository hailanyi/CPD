import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import box_coder_utils, common_utils, loss_utils
from ..model_utils.model_nms_utils import class_agnostic_nms
from .target_assigner.proposal_target_layer import ProposalTargetLayer, ProposalTargetLayerT
from ...utils.odiou_loss import odiou_3D
from ...utils.bbloss import bb_loss

import time
import copy

class RoIHeadTemplate(nn.Module):
    def __init__(self, num_class,num_frames, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.box_coder = getattr(box_coder_utils, self.model_cfg.TARGET_CONFIG.BOX_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', {})
        )
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = {}
        self.num_frames = num_frames
        self.iter=1

    def build_losses(self, losses_cfg):
        self.add_module(
            'reg_loss_func',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )


    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    @torch.no_grad()
    def proposal_layer(self, batch_dict, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        if batch_dict.get('rois', None) is not None:
            return batch_dict

        batch_size = batch_dict['batch_size']
        batch_box_preds = batch_dict['batch_box_preds']
        batch_cls_preds = batch_dict['batch_cls_preds']


        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)


        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]

            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)

            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )

            rois[index, :len(selected), :] = box_preds[selected]

            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]

        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores

        batch_dict['roi_labels'] = roi_labels + 1
        batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 else False
        batch_dict.pop('batch_index', None)
        return batch_dict

    def assign_targets(self, batch_dict,ind=''):
        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            targets_dict = self.proposal_target_layer.forward(batch_dict,ind)

        rois = targets_dict['rois'+ind]  # (B, N, 7 + C)
        gt_of_rois = targets_dict['gt_of_rois'+ind]  # (B, N, 7 + C + 1)
        targets_dict['gt_of_rois_src'+ind] = gt_of_rois.clone().detach()

        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # transfer LiDAR coords to local coords
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'+ind] = gt_of_rois
        return targets_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
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
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

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
                tb_dict['rcnn_loss_corner'] = loss_corner.item()
        else:
            raise NotImplementedError


        return rcnn_loss_reg, tb_dict


    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)

        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def smooth_l1_loss(self, diff, beta=1.0 / 9.0):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def proto_loss(self, reg0, reg1_in, cls0, cls1_in, label):

        reg1 = reg1_in.clone().detach()
        cls1 = cls1_in.clone().detach()

        cls_valid_mask = (label >= 0).float()

        l1 = self.smooth_l1_loss(reg1-reg0)
        l1 = l1.mean(dim=-1)
        l2 = self.smooth_l1_loss(torch.sigmoid(cls1)-torch.sigmoid(cls0))

        l = l1+l2

        loss = (l * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

        return loss

    def proto_loss2(self, reg0, reg1_in, label):

        reg1 = reg1_in.clone().detach()

        cls_valid_mask = (label >= 0).float()

        l1 = self.smooth_l1_loss(reg1-reg0)
        l1 = l1.mean(dim=-1)

        loss = (l1 * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

        return loss

    def proto_loss3(self, targets_dict0, targets_dict1):


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

        if len(gt_boxes3d_ct[fg_mask]) == 0:
            b_loss0 = 0
        else:
            b_loss0 = bb_loss(batch_box_preds0[fg_mask], gt_boxes3d_ct[fg_mask]).sum()
            b_loss0 = b_loss0 / (fg_mask.sum() + 1)


        rcnn_reg1 = targets_dict1['rcnn_reg']  # (rcnn_batch_size, C)
        rois1 = targets_dict1['rois'].clone().view(-1, code_size)[:, 0:7]
        rois1[:, 0:3] = 0
        rois1[:, 6] = 0

        batch_box_preds1 = self.box_coder.decode_torch(rcnn_reg1, rois1).view(-1, code_size)
        batch_box_preds1 = batch_box_preds1.clone().detach()


        if len(gt_boxes3d_ct[fg_mask]) == 0:
            b_loss1 = 0
        else:
            b_loss1 = bb_loss(batch_box_preds0[fg_mask], batch_box_preds1[fg_mask]).sum()
            b_loss1 = b_loss1 / (fg_mask.sum() + 1)


        begin_weight = 0.00001
        max_iter = 5000
        end_weight = 0.2

        if self.iter > max_iter:
            self.iter = max_iter
        this_weight = (self.iter / max_iter) * (end_weight - begin_weight) + begin_weight
        self.iter += 1

        b_loss1*=this_weight

        cls0 = targets_dict0['rcnn_cls'].view(-1)
        cls1_in = targets_dict1['rcnn_cls'].view(-1)
        label = targets_dict0['rcnn_cls_labels'].view(-1)
        reg0 = targets_dict0['shared_features']
        reg1_in = targets_dict1['shared_features']
        reg1 = reg1_in.clone().detach()
        cls1 = cls1_in.clone().detach()

        l = torch.cosine_similarity(reg0, reg1,dim=-1)

        cls_valid_mask = (label >= 0).float()

        loss_sum = (l * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

        loss_sum = loss_sum*this_weight
        b_loss1 = b_loss1*this_weight

        final = b_loss0+b_loss1+loss_sum

        return final

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

        proto_l += self.proto_loss3(self.forward_ret_dict['targets_dict0'], self.forward_ret_dict['targets_dict1'])

        # proto_l = self.proto_loss2(self.forward_ret_dict['targets_dict0']['shared_features'].view(-1),
        #                           self.forward_ret_dict['targets_dict1']['shared_features'].view(-1),
        #                           self.forward_ret_dict['targets_dict0']['rcnn_cls_labels'].view(-1))

        rcnn_loss+=(proto_l*loss_cfgs.LOSS_WEIGHTS['rcnn_proto_weight'])

        tb_dict['rcnn_loss'] = rcnn_loss.item()

        return rcnn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)


        return batch_cls_preds, batch_box_preds


class CascadeRoIHeadTemplate(nn.Module):
    def __init__(self, num_class,num_frames, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.box_coder = getattr(box_coder_utils, self.model_cfg.TARGET_CONFIG.BOX_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', {})
        )
        
        self.proposal_target_layers = []
        for i in range(self.model_cfg.STAGES):
            proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG['STAGE'+str(i)])
            self.proposal_target_layers.append(proposal_target_layer)
            
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = {}
        self.num_frames = num_frames

    def build_losses(self, losses_cfg):
        self.add_module(
            'reg_loss_func',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'od_loss_func',
            odiou_3D()
        )

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    @torch.no_grad()
    def proposal_layer(self, batch_dict, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """

        if batch_dict.get('rois', None) is not None:
            batch_dict['cls_preds_normalized'] = False
            return batch_dict

        batch_size = batch_dict['batch_size']
        batch_box_preds = batch_dict['batch_box_preds']
        batch_cls_preds = batch_dict['batch_cls_preds']

        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)


        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]

            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)

            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )

            rois[index, :len(selected), :] = box_preds[selected]

            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]

        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores

        batch_dict['roi_labels'] = roi_labels + 1
        batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 else False
        batch_dict.pop('batch_index', None)
        return batch_dict

    def assign_targets(self, batch_dict,stage_id):
        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            targets_dict = self.proposal_target_layers[stage_id].forward(batch_dict)

        rois = targets_dict['rois']  # (B, N, 7 + C)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # transfer LiDAR coords to local coords
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'].clone()[..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
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
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

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
                tb_dict['rcnn_loss_corner'] = loss_corner.item()
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)

        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_single_semi_loss(self, preds, semi):
        semi_scores = semi['rcnn_cls']
        semi_scores = torch.sigmoid(semi_scores).view(-1)
        rcnn_cls = preds['rcnn_cls']
        rcnn_cls = torch.sigmoid(rcnn_cls).view(-1)

        loss_cls = (semi_scores-rcnn_cls)**2#semi_scores*torch.log(semi_scores/rcnn_cls)+(1-semi_scores)*torch.log((1-semi_scores)/(1-rcnn_cls))
        loss_cls = torch.mean(loss_cls)

        tb_dict = {'KL_dis_semi_loss:', loss_cls.item()}

        return loss_cls, tb_dict

    def get_od_loss(self, forward_ret_dict):
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        code_size = self.box_coder.code_size
        shape = forward_ret_dict['gt_of_rois'].shape
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'].clone().view(shape[0]*shape[1], -1)[:, 0:7]
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        rois = forward_ret_dict['rois'].clone().view(-1, code_size)[:, 0:7]
        rois[:, 0:3] = 0
        rois[:, 6] = 0

        batch_box_preds = self.box_coder.decode_torch(rcnn_reg, rois).view(-1, code_size)

        fg_mask = (reg_valid_mask > 0)

        if len(gt_boxes3d_ct[fg_mask]) == 0:
            return 0

        od_loss = self.od_loss_func(gt_boxes3d_ct[fg_mask], batch_box_preds[fg_mask], 1, 1)

        od_loss = od_loss/(fg_mask.sum()+1)

        return od_loss



    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        for i in range(self.model_cfg.STAGES):
            rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict['targets_dict'+str(i)])
            rcnn_loss += rcnn_loss_cls

            rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict['targets_dict'+str(i)])
            rcnn_loss += rcnn_loss_reg

            #od_loss = self.get_od_loss(self.forward_ret_dict['targets_dict'+str(i)])
            #rcnn_loss += od_loss

        tb_dict['rcnn_loss'] = rcnn_loss.item()

        return rcnn_loss, tb_dict

    def get_semi_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_semi_loss = 0
        for i in range(self.model_cfg.STAGES):
            rcnn_loss, cls_tb_dict = self.get_single_semi_loss(self.forward_ret_dict['targets_dict'+str(i)],self.forward_ret_dict['semi_dict'+str(i)])
            rcnn_semi_loss += rcnn_loss
        tb_dict['rcnn_semi_loss'] = rcnn_semi_loss.item()
        return rcnn_semi_loss,tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)


        return batch_cls_preds, batch_box_preds


class ToIHeadTemplate(nn.Module):
    def __init__(self, num_class,num_frames, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.box_coder = getattr(box_coder_utils, self.model_cfg.TARGET_CONFIG.BOX_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', {})
        )
        self.proposal_target_layer = ProposalTargetLayerT(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None
        self.num_frames = num_frames

    def build_losses(self, losses_cfg):
        self.add_module(
            'reg_loss_func',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    def generate_tracklets(self, batch_dict, pc_range, voxel_size, mode = 'chain_bbs', map_stride=8): # 'chain_mots'

        batch_size = batch_dict['batch_size']

        if 'temporal_features' in batch_dict:
            spatial_size = batch_dict['temporal_features'].shape
        else:
            spatial_size = batch_dict['spatial_features'].shape

        batch_box_preds = batch_dict['batch_box_preds'].view(batch_size,spatial_size[-2],spatial_size[-1],-1,9)
        anchor_num = batch_box_preds.shape[-2]

        num_frames = self.num_frames

        tracklets = batch_box_preds.new_zeros((batch_size,spatial_size[-2],spatial_size[-1],anchor_num,num_frames*7+2))

        tracklets[...,0:9] = batch_box_preds[...,0:9]

        for i in range(1,self.num_frames):
            pre_i = (i-1)*7

            track_i = i*7

            pre_box = tracklets[..., pre_i:pre_i+9].view(batch_size, spatial_size[-2], spatial_size[-1], -1, 9)
            pre_mot = pre_box[..., 7:9]
            pre_box = pre_box[..., 0:7]

            this_box = batch_dict['batch_box_preds'+str(-i)].view(batch_size, spatial_size[-2], spatial_size[-1], -1, 9)

            pred_position = pre_box[...,0:2]+pre_mot[...,0:2]

            coords_x = (pred_position[..., 0:1] - pc_range[0]) // (voxel_size[0]*map_stride)
            coords_y = (pred_position[..., 1:2] - pc_range[1]) // (voxel_size[1]*map_stride)

            coords_x = coords_x.long()
            coords_y = coords_y.long()

            all_batch=[]

            for b_i in range(batch_size):

                each_batch = []

                for a_i in range(anchor_num):

                    boxes = this_box[b_i,:,:,a_i,:]

                    c_x = coords_x[b_i,:,:,a_i,0].reshape(-1)
                    c_y = coords_y[b_i,:,:,a_i,0].reshape(-1)

                    c_x = torch.clamp(c_x,min=0,max = spatial_size[-1]-1)
                    c_y = torch.clamp(c_y, min=0, max= spatial_size[-2]-1)

                    each_batch.append(boxes[c_y,c_x])

                batch_bbs = torch.stack(each_batch).permute(1,0,2).view(spatial_size[-2], spatial_size[-1],anchor_num,9)
                all_batch.append(batch_bbs)
            all_batch = torch.stack(all_batch)

            if mode == 'chain_bbs':
                tracklets[...,track_i:track_i+9] = all_batch[...,0:9]

            elif mode == 'chain_mots':
                tracklets[..., track_i:track_i + 9] = all_batch[..., 0:9]

                tracklets[..., track_i+2:track_i + 7] = pre_box[..., 2:7]

        tracklets = tracklets.view(batch_size, -1, num_frames*7+2)
        batch_dict['batch_tracks_preds'] = tracklets[...,0:num_frames*7]

        gt_tracklets = batch_dict['gt_tracklets']

        new_gt_tracklets = gt_tracklets.new_zeros((gt_tracklets.shape[0],gt_tracklets.shape[1],num_frames*7))
        new_gt_tracklets[...,0:7] = gt_tracklets[...,0:7]

        for i in range(1,num_frames):
            new_gt_tracklets[..., i * 7 + 3:i * 7 + 6] = gt_tracklets[..., 3:6]
            new_gt_tracklets[..., i * 7 + 6] = gt_tracklets[..., 7+(i-1)*4+3]
            new_gt_tracklets[..., i * 7: i * 7+3] = gt_tracklets[..., 7+(i-1)*4:7+(i-1)*4+3]
        batch_dict['gt_tracklets']=torch.cat([new_gt_tracklets,gt_tracklets[...,-1:]],-1)

        num_bbs_in_tracklets = batch_dict['num_bbs_in_tracklets']

        gt_bbs_mask = gt_tracklets.new_zeros((gt_tracklets.shape[0],gt_tracklets.shape[1],num_frames))

        for i in range(num_frames):
            gt_bbs_mask[...,i:i+1] = num_bbs_in_tracklets>=(i+1)

        batch_dict['gt_bbs_mask'] = gt_bbs_mask


        return batch_dict


    @torch.no_grad()
    def proposal_layer(self, batch_dict, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        batch_size = batch_dict['batch_size']
        batch_box_preds = batch_dict['batch_tracks_preds']
        batch_cls_preds = batch_dict['batch_cls_preds']

        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)


        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]

            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)

            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )

            rois[index, :len(selected), :] = box_preds[selected]

            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]

        batch_dict['rois'] = rois

        batch_dict['roi_scores'] = roi_scores

        batch_dict['roi_labels'] = roi_labels + 1
        batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 else False
        batch_dict.pop('batch_index', None)
        return batch_dict

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            targets_dict = self.proposal_target_layer.forward(batch_dict)

        rois = targets_dict['rois']  # (B, N, 7 + C)
        all_gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
        targets_dict['gt_of_rois_src'] = all_gt_of_rois.clone().detach()

        for frame_id in range(self.num_frames):
            this_rois = rois[..., frame_id * 7:frame_id * 7 + 7]
            gt_of_rois = all_gt_of_rois[..., frame_id * 7:frame_id * 7 + 7]

            # canonical transformation
            roi_center = this_rois[:, :, 0:3]
            roi_ry = this_rois[:, :, 6] % (2 * np.pi)
            gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
            gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

            # transfer LiDAR coords to local coords
            gt_of_rois = common_utils.rotate_points_along_z(
                points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
            ).view(batch_size, -1, gt_of_rois.shape[-1])

            # flip orientation if rois have opposite orientation
            heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
            heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (
                        2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
            flag = heading_label > np.pi
            heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
            heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

            gt_of_rois[:, :, 6] = heading_label

            all_gt_of_rois[..., frame_id * 7:frame_id * 7 + 7] = gt_of_rois[..., 0:7]

        targets_dict['gt_of_rois'] = all_gt_of_rois

        return targets_dict

    def get_box_reg_layer_loss(self, forward_ret_dict,frame_id=0):

        if frame_id == 0:
            frame_id_str=''
        else:
            frame_id_str=str(-frame_id)

        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        reg_valid_mask*= forward_ret_dict['gt_bbs_mask'][...,frame_id].view(-1).long()


        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., frame_id*7:frame_id*7+7]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., frame_id*7:frame_id*7+7].view(-1, 7)
        rcnn_reg = forward_ret_dict['rcnn_reg'+frame_id_str]  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois'][...,frame_id*7:frame_id*7+7]
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
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
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'+frame_id_str] = rcnn_loss_reg.item()

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
                tb_dict['rcnn_loss_corner'+frame_id_str] = loss_corner.item()
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)

        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']

        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict


    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0

        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        frame_weights=self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['frame_weights']

        all_reg_loss = 0

        for i in range(self.num_frames):

            rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict,i)
            all_reg_loss += rcnn_loss_reg*frame_weights[i]

        rcnn_loss+=all_reg_loss

        tb_dict['rcnn_loss'] = rcnn_loss.item()

        return rcnn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)


        return batch_cls_preds, batch_box_preds

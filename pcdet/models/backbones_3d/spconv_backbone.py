from functools import partial
from ...utils.spconv_utils import replace_feature, spconv
import torch.nn as nn
import numpy as np
import torch
from pcdet.datasets.augmentor.X_transform import X_TRAIN
import time
import random

from spconv.pytorch.utils import PointToVoxel, gather_features_by_pc_voxel_id


def d2_max_pool(index, features, batch_size = 2 ):
    pooled_feat = features.clone()
    for i in range(batch_size):
        mask = index[:, 0] == i
        this_feat = features[mask]
        this_index = index[mask]
        this_index = this_index[:, 1:]

        features_num = features.shape[-1]
        zeros = torch.zeros(size=(this_index.shape[0], 1)).to(this_index.device)
        new_features = torch.cat([this_index, zeros, this_feat], -1)
        device = this_feat.device

        gen = PointToVoxel(vsize_xyz=[1, 1, 1],
                           coors_range_xyz=[0, 0, 0, 1400, 1400, 2],
                           num_point_features=new_features.shape[-1],
                           max_num_voxels=100000,
                           max_num_points_per_voxel=10,
                           device=device)

        voxels_th, indices_th, num_p_in_vx_th, pc_voxel_id = gen.generate_voxel_with_id(new_features, empty_mean=True)

        max_feat = voxels_th[:, :, :].max(dim=1, keepdim=False)[0]
        max_feat = max_feat[:, -features_num:]

        pc_features = gather_features_by_pc_voxel_id(max_feat, pc_voxel_id)

        pooled_feat[mask] = pc_features

    return pooled_feat


def index2points(indices, pts_range=[0, -40, -3, 70.4, 40, 1], voxel_size=[0.05, 0.05, 0.05], stride=8):

    voxel_size = np.array(voxel_size)*stride
    min_x = pts_range[0] + voxel_size[0] / 2
    min_y = pts_range[1] + voxel_size[1] / 2
    min_z = pts_range[2] + voxel_size[2] / 2

    new_indices = indices.clone().float()
    indices_float = indices.clone().float()
    new_indices[:, 1] = indices_float[:, 3] * voxel_size[0] + min_x
    new_indices[:, 2] = indices_float[:, 2] * voxel_size[1] + min_y
    new_indices[:, 3] = indices_float[:, 1] * voxel_size[2] + min_z

    return new_indices

def index2uv3d(indices, batch_size, calib, stride, x_trans_train, trans_param):
    new_uv = indices.clone().int()
    for b_i in range(batch_size):
        cur_in = indices[indices[:, 0]==b_i]
        cur_pts = index2points(cur_in, stride=stride)
        if trans_param is not None:
            transed = x_trans_train.backward_with_param({'points': cur_pts[:, 1:4],
                                                              'transform_param': trans_param[b_i]})
            cur_pts = transed['points'].cpu().numpy()
        else:
            cur_pts = cur_pts[:, 1:4].cpu().numpy()

        pts_rect = calib[b_i].lidar_to_rect(cur_pts[:, 0:3])
        pts_img, pts_rect_depth = calib[b_i].rect_to_img(pts_rect)
        pts_img = pts_img.astype(np.int32)
        pts_img = torch.from_numpy(pts_img).to(new_uv.device)
        new_uv[indices[:, 0]==b_i, 2:4] = pts_img

    new_uv[:, 1] = 1
    new_uv[:, 2] = torch.clamp(new_uv[:, 2], min=0, max = 1400)//stride
    new_uv[:, 3] = torch.clamp(new_uv[:, 3], min=0, max = 600)//stride
    return new_uv


def index2uv(indices, batch_size, calib, stride, x_trans_train, trans_param):
    new_uv = indices.new(size = (indices.shape[0], 3))
    depth = indices.new(size = (indices.shape[0], 1)).float()
    for b_i in range(batch_size):
        cur_in = indices[indices[:, 0]==b_i]
        cur_pts = index2points(cur_in, stride=stride)
        if trans_param is not None:
            transed = x_trans_train.backward_with_param({'points': cur_pts[:, 1:4],
                                                              'transform_param': trans_param[b_i]})
            cur_pts = transed['points']#.cpu().numpy()
        else:
            cur_pts = cur_pts[:, 1:4]#.cpu().numpy()

        pts_rect = calib[b_i].lidar_to_rect_cuda(cur_pts[:, 0:3])
        pts_img, pts_rect_depth = calib[b_i].rect_to_img_cuda(pts_rect)

        pts_img = pts_img.int()
        #pts_img = torch.from_numpy(pts_img).to(new_uv.device)
        new_uv[indices[:, 0]==b_i, 1:3] = pts_img
        #pts_rect_depth = torch.from_numpy(pts_rect_depth).to(new_uv.device).float()
        depth[indices[:, 0]==b_i, 0] = pts_rect_depth[:]
    new_uv[:, 0] = indices[:, 0]
    new_uv[:, 1] = torch.clamp(new_uv[:, 1], min=0, max = 1400-1)//stride
    new_uv[:, 2] = torch.clamp(new_uv[:, 2], min=0, max = 600-1)//stride

    return new_uv, depth

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        relu = nn.ReLU()
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
        relu =nn.ReLU(inplace=True)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        relu = nn.ReLU()
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        relu,
    )

    return m

def post_act_block2d(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        relu = nn.ReLU()
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
        relu =nn.ReLU(inplace=True)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        relu = nn.ReLU()
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        relu,
    )

    return m

def voxel_sampling(sparse_t, coords, rat = 0.15):
    if rat == 0:
        return 

    len = sparse_t.features.shape[0]
    randoms = np.random.permutation(len)
    randoms = torch.from_numpy(randoms[0:int(len*(1-rat))]).to(sparse_t.features.device)
    
    sparse_t = replace_feature(sparse_t, sparse_t.features[randoms])
    sparse_t.indices = sparse_t.indices[randoms]

class BasicBlock(spconv.SparseModule):

    def __init__(self, inplanes, planes,  norm_fn=None, stride=2,  padding=1,  indice_key=None):
        super(BasicBlock, self).__init__()

        assert norm_fn is not None

        block = post_act_block
        self.stride = stride
        if stride >1:
            self.down_conv = block(inplanes,
                                    planes,
                                    3,
                                    norm_fn=norm_fn,
                                    stride=2,
                                    padding=padding,
                                    indice_key=('sp' + indice_key),
                                    conv_type='spconv')
        if stride >1:
            conv_in = planes
        else:
            conv_in = inplanes

        self.conv1 = block(conv_in,
                              planes // 2,
                              3,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm1' + indice_key))
        self.conv2 = block(planes//2,
                              planes // 2,
                              3,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm2' + indice_key))

        self.conv3 = block(planes//2,
                              planes // 2,
                              3,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm3' + indice_key))
        self.conv4 = block(planes//2,
                              planes // 2,
                              3,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm4' + indice_key))


    def forward(self, x):

        if self.stride>1:
            x = self.down_conv(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        out = replace_feature(x2, torch.cat([x1.features, x4.features],-1))

        return out

class TeVoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.return_num_features_as_dict = model_cfg.RETURN_NUM_FEATURES_AS_DICT
        self.out_features=model_cfg.OUT_FEATURES

        num_filters = model_cfg.NUM_FILTERS

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(num_filters[0]),
            nn.ReLU(),
        )
        block = post_act_block
        self.conv1 = spconv.SparseSequential(
            block(num_filters[0], num_filters[0], 3, norm_fn=norm_fn, padding=1, indice_key='conv1'),
        )
        self.conv2 = BasicBlock(num_filters[0], num_filters[1], norm_fn=norm_fn,  indice_key='conv2')
        self.conv3 = BasicBlock(num_filters[1], num_filters[2], norm_fn=norm_fn,  indice_key='conv3')
        self.conv4 = BasicBlock(num_filters[2], num_filters[3], norm_fn=norm_fn,  padding=(0, 1, 1),  indice_key='conv4')


        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(num_filters[3], self.out_features, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_features),
            nn.ReLU(),
        )
        if self.model_cfg.get('MM', False):
            self.conv_input_2 = spconv.SparseSequential(
                spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1_2'),
                norm_fn(num_filters[0]),
                nn.ReLU(),
            )

            self.conv1_2 = spconv.SparseSequential(
                block(num_filters[0], num_filters[0], 3, norm_fn=norm_fn, padding=1, indice_key='conv1_2'),
            )
            self.conv2_2 = BasicBlock(num_filters[0], num_filters[1], norm_fn=norm_fn, indice_key='conv2_2')
            self.conv3_2 = BasicBlock(num_filters[1], num_filters[2], norm_fn=norm_fn, indice_key='conv3_2')
            self.conv4_2 = BasicBlock(num_filters[2], num_filters[3], norm_fn=norm_fn, padding=(0, 1, 1),  indice_key='conv4_2')


        self.num_point_features = self.out_features

        if self.return_num_features_as_dict:
            num_point_features = {}
            num_point_features.update({
                'x_conv1': num_filters[0],
                'x_conv2': num_filters[1],
                'x_conv3': num_filters[2],
                'x_conv4': num_filters[3],
            })
            self.num_point_features = num_point_features


    def decompose_tensor(self, tensor, i, batch_size):
        input_shape = tensor.spatial_shape[2]
        begin_shape_ids = i * (input_shape // 4)
        end_shape_ids = (i + 1) * (input_shape // 4)
        x_conv3_features = tensor.features
        x_conv3_coords = tensor.indices

        mask = (begin_shape_ids < x_conv3_coords[:, 3]) & (x_conv3_coords[:, 3] < end_shape_ids)
        this_conv3_feat = x_conv3_features[mask]
        this_conv3_coords = x_conv3_coords[mask]
        this_conv3_coords[:, 3] -= i * (input_shape // 4)
        this_shape = [tensor.spatial_shape[0], tensor.spatial_shape[1], tensor.spatial_shape[2] // 4]

        this_conv3_tensor = spconv.SparseConvTensor(
            features=this_conv3_feat,
            indices=this_conv3_coords.int(),
            spatial_shape=this_shape,
            batch_size=batch_size
        )
        return this_conv3_tensor

    def forward_test(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """

        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            rot_num = trans_param.shape[1]
        else:
            rot_num = 1

        all_lidar_feat = []
        all_lidar_coords = []

        new_shape = [self.sparse_shape[0], self.sparse_shape[1], self.sparse_shape[2] * 4]

        for i in range(rot_num):
            if i==0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)

            voxel_features, voxel_coords = batch_dict['voxel_features'+rot_num_id], batch_dict['voxel_coords'+rot_num_id]

            all_lidar_feat.append(voxel_features)
            new_coord = voxel_coords.clone()
            new_coord[:, 3] += i*self.sparse_shape[2]
            all_lidar_coords.append(new_coord)
        batch_size = batch_dict['batch_size']

        all_lidar_feat = torch.cat(all_lidar_feat, 0)
        all_lidar_coords = torch.cat(all_lidar_coords)

        input_sp_tensor = spconv.SparseConvTensor(
            features=all_lidar_feat,
            indices=all_lidar_coords.int(),
            spatial_shape=new_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        for i in range(rot_num):
            if i==0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)

            this_conv3 = self.decompose_tensor(x_conv3, i, batch_size)
            this_conv4 = self.decompose_tensor(x_conv4, i, batch_size)
            this_out = self.decompose_tensor(out, i, batch_size)

            batch_dict.update({
                'encoded_spconv_tensor'+rot_num_id: this_out,
                'encoded_spconv_tensor_stride'+rot_num_id: 8,
            })
            batch_dict.update({
                'multi_scale_3d_features'+rot_num_id: {
                    'x_conv1': None,
                    'x_conv2': None,
                    'x_conv3': this_conv3,
                    'x_conv4': this_conv4,
                },
                'multi_scale_3d_strides'+rot_num_id: {
                    'x_conv1': 1,
                    'x_conv2': 2,
                    'x_conv3': 4,
                    'x_conv4': 8,
                }
            })


        if self.model_cfg.get('MM', False):
            all_mm_feat = []
            all_mm_coords = []
            for i in range(rot_num):
                if i == 0:
                    rot_num_id = ''
                else:
                    rot_num_id = str(i)

                newvoxel_features, newvoxel_coords = batch_dict['voxel_features_mm'+rot_num_id], batch_dict['voxel_coords_mm'+rot_num_id]

                all_mm_feat.append(newvoxel_features)
                new_mm_coord = newvoxel_coords.clone()
                new_mm_coord[:, 3] += i * self.sparse_shape[2]
                all_mm_coords.append(new_mm_coord)
            all_mm_feat = torch.cat(all_mm_feat, 0)
            all_mm_coords = torch.cat(all_mm_coords)

            newinput_sp_tensor = spconv.SparseConvTensor(
                features=all_mm_feat,
                indices=all_mm_coords.int(),
                spatial_shape=new_shape,
                batch_size=batch_size
            )

            newx = self.conv_input_2(newinput_sp_tensor)

            newx_conv1 = self.conv1_2(newx)
            newx_conv2 = self.conv2_2(newx_conv1)
            newx_conv3 = self.conv3_2(newx_conv2)
            newx_conv4 = self.conv4_2(newx_conv3)

            for i in range(rot_num):
                if i == 0:
                    rot_num_id = ''
                else:
                    rot_num_id = str(i)

                this_conv3 = self.decompose_tensor(newx_conv3, i, batch_size)
                this_conv4 = self.decompose_tensor(newx_conv4, i, batch_size)
                batch_dict.update({
                    'encoded_spconv_tensor_stride_mm'+rot_num_id: 8
                })
                batch_dict.update({
                    'multi_scale_3d_features_mm'+rot_num_id: {
                        'x_conv1': None,
                        'x_conv2': None,
                        'x_conv3': this_conv3,
                        'x_conv4': this_conv4,
                    },
                    'multi_scale_3d_strides'+rot_num_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })

        return batch_dict

    def forward_train(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            rot_num = trans_param.shape[1]
        else:
            rot_num = 1


        for i in range(rot_num):
            if i==0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)

            voxel_features, voxel_coords = batch_dict['voxel_features'+rot_num_id], batch_dict['voxel_coords'+rot_num_id]

            batch_size = batch_dict['batch_size']
            input_sp_tensor = spconv.SparseConvTensor(
                features=voxel_features,
                indices=voxel_coords.int(),
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )
            x = self.conv_input(input_sp_tensor)

            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)

            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)

            batch_dict.update({
                'encoded_spconv_tensor'+rot_num_id: out,
                'encoded_spconv_tensor_stride'+rot_num_id: 8,
            })
            batch_dict.update({
                'multi_scale_3d_features'+rot_num_id: {
                    'x_conv1': x_conv1,
                    'x_conv2': x_conv2,
                    'x_conv3': x_conv3,
                    'x_conv4': x_conv4,
                },
                'multi_scale_3d_strides'+rot_num_id: {
                    'x_conv1': 1,
                    'x_conv2': 2,
                    'x_conv3': 4,
                    'x_conv4': 8,
                }
            })

            if self.model_cfg.get('MM', False):
                newvoxel_features, newvoxel_coords = batch_dict['voxel_features_mm'+rot_num_id], batch_dict['voxel_coords_mm'+rot_num_id]

                newinput_sp_tensor = spconv.SparseConvTensor(
                    features=newvoxel_features,
                    indices=newvoxel_coords.int(),
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_size
                )
                newx = self.conv_input_2(newinput_sp_tensor)

                newx_conv1 = self.conv1_2(newx)
                newx_conv2 = self.conv2_2(newx_conv1)
                newx_conv3 = self.conv3_2(newx_conv2)
                newx_conv4 = self.conv4_2(newx_conv3)

                # for detection head
                # [200, 176, 5] -> [200, 176, 2]
                #newout = self.conv_out(newx_conv4)

                batch_dict.update({
                    #'encoded_spconv_tensor_mm': newout,
                    'encoded_spconv_tensor_stride_mm'+rot_num_id: 8
                })
                batch_dict.update({
                    'multi_scale_3d_features_mm'+rot_num_id: {
                        'x_conv1': newx_conv1,
                        'x_conv2': newx_conv2,
                        'x_conv3': newx_conv3,
                        'x_conv4': newx_conv4,
                    },
                    'multi_scale_3d_strides'+rot_num_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })

        return batch_dict

    def forward(self, batch_dict):
        if self.training:
            return self.forward_train(batch_dict)
        else:
            return self.forward_test(batch_dict)

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

class VirConvBlock(nn.Module):
    def __init__(self, input_c=16, output_c=16, stride=1, padding=1, indice_key='vir1', conv_depth=False):
        super(VirConvBlock, self).__init__()
        self.stride = stride
        block = post_act_block
        block2d = post_act_block2d
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.conv_depth = conv_depth

        if self.stride > 1:
            self.down_layer = block(input_c,
                                    output_c,
                                    3,
                                    norm_fn=norm_fn,
                                    stride=stride,
                                    padding=padding,
                                    indice_key=('sp' + indice_key),
                                    conv_type='spconv')
        c1 = input_c

        if self.stride > 1:
            c1 = output_c
        if self.conv_depth:
            c1 += 4

        c2 = output_c

        self.d3_conv1 = block(c1,
                              c2 // 2,
                              3,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm1' + indice_key))
        self.d2_conv1 = block2d(c2 // 2,
                                c2 // 2,
                                3,
                                norm_fn=norm_fn,
                                padding=1,
                                indice_key=('subm3' + indice_key))

        self.d3_conv2 = block(c2 // 2,
                              c2 // 2,
                              3,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm2' + indice_key))
        self.d2_conv2 = block2d(c2 // 2,
                                c2 // 2,
                                3,
                                norm_fn=norm_fn,
                                padding=1,
                                indice_key=('subm4' + indice_key))

    def forward(self, sp_tensor, batch_size, calib, stride, x_trans_train, trans_param):

        if self.stride > 1:
            sp_tensor = self.down_layer(sp_tensor)

        d3_feat1 = self.d3_conv1(sp_tensor)
        d3_feat2 = self.d3_conv2(d3_feat1)

        uv_coords, depth = index2uv(d3_feat2.indices, batch_size, calib, stride, x_trans_train, trans_param)

        pool_feat = d2_max_pool(uv_coords,d3_feat2.features, batch_size)

        d2_sp_tensor1 = spconv.SparseConvTensor(
            features=pool_feat,
            indices=uv_coords.int(),
            spatial_shape=[1600, 600],
            batch_size=batch_size
        )

        d2_feat1 = self.d2_conv1(d2_sp_tensor1)
        d2_feat2 = self.d2_conv2(d2_feat1)

        d3_feat3 = replace_feature(d3_feat2, torch.cat([d3_feat2.features, d2_feat2.features], -1))

        return d3_feat3

class VirConv8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, num_frames, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_frames = num_frames
        self.x_trans_train = X_TRAIN()

        self.return_num_features_as_dict = model_cfg.RETURN_NUM_FEATURES_AS_DICT
        self.out_features=model_cfg.OUT_FEATURES

        num_filters = model_cfg.NUM_FILTERS

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(num_filters[0]),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(num_filters[0], num_filters[0], 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(num_filters[0], num_filters[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(num_filters[1], num_filters[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(num_filters[2], num_filters[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(num_filters[3], self.out_features, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_features),
            nn.ReLU(),
        )
        if self.model_cfg.get('MM', False):
            self.vir_conv1 = VirConvBlock(input_channels, num_filters[0], stride=1, indice_key='vir1')
            self.vir_conv2 = VirConvBlock(num_filters[0], num_filters[1], stride=2, indice_key='vir2')
            self.vir_conv3 = VirConvBlock(num_filters[1], num_filters[2], stride=2, indice_key='vir3')
            self.vir_conv4 = VirConvBlock(num_filters[2], num_filters[3], stride=2, padding=(0,1,1), indice_key='vir4')

        self.num_point_features = self.out_features

        if self.return_num_features_as_dict:
            num_point_features = {}
            num_point_features.update({
                'x_conv1': num_filters[0],
                'x_conv2': num_filters[1],
                'x_conv3': num_filters[2],
                'x_conv4': num_filters[3],
            })
            self.num_point_features = num_point_features
        #for child in self.children():
        #    for param in child.parameters():
        #        param.requires_grad = False

    def decompose_tensor(self, tensor, i, batch_size):
        input_shape = tensor.spatial_shape[2]
        begin_shape_ids = i * (input_shape // 4)
        end_shape_ids = (i + 1) * (input_shape // 4)
        x_conv3_features = tensor.features
        x_conv3_coords = tensor.indices

        mask = (begin_shape_ids < x_conv3_coords[:, 3]) & (x_conv3_coords[:, 3] < end_shape_ids)
        this_conv3_feat = x_conv3_features[mask]
        this_conv3_coords = x_conv3_coords[mask]
        this_conv3_coords[:, 3] -= i * (input_shape // 4)
        this_shape = [tensor.spatial_shape[0], tensor.spatial_shape[1], tensor.spatial_shape[2] // 4]

        this_conv3_tensor = spconv.SparseConvTensor(
            features=this_conv3_feat,
            indices=this_conv3_coords.int(),
            spatial_shape=this_shape,
            batch_size=batch_size
        )
        return this_conv3_tensor

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            stages = trans_param.shape[1]
        else:
            stages = 1


        batch_size = batch_dict['batch_size']
        all_lidar_feat = []
        all_lidar_coords = []

        new_shape = [self.sparse_shape[0], self.sparse_shape[1], self.sparse_shape[2] * 4]

        if self.training:
            for i in range(stages):
                if i == 0:
                    stage_id = ''
                else:
                    stage_id = str(i)

                voxel_features, voxel_coords = batch_dict['voxel_features' + stage_id], batch_dict[
                    'voxel_coords' + stage_id]

                batch_size = batch_dict['batch_size']
                input_sp_tensor = spconv.SparseConvTensor(
                    features=voxel_features,
                    indices=voxel_coords.int(),
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_size
                )
                x = self.conv_input(input_sp_tensor)

                x_conv1 = self.conv1(x)
                x_conv2 = self.conv2(x_conv1)
                x_conv3 = self.conv3(x_conv2)
                x_conv4 = self.conv4(x_conv3)

                # for detection head
                # [200, 176, 5] -> [200, 176, 2]
                out = self.conv_out(x_conv4)

                batch_dict.update({
                    'encoded_spconv_tensor' + stage_id: out,
                    'encoded_spconv_tensor_stride' + stage_id: 8,
                })
                batch_dict.update({
                    'multi_scale_3d_features' + stage_id: {
                        'x_conv1': x_conv1,
                        'x_conv2': x_conv2,
                        'x_conv3': x_conv3,
                        'x_conv4': x_conv4,
                    },
                    'multi_scale_3d_strides' + stage_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })
        else:
            for i in range(stages):
                if i==0:
                    stage_id = ''
                else:
                    stage_id = str(i)

                voxel_features, voxel_coords = batch_dict['voxel_features'+stage_id], batch_dict['voxel_coords'+stage_id]

                all_lidar_feat.append(voxel_features)
                new_coord = voxel_coords.clone()
                new_coord[:, 3] += i*self.sparse_shape[2]
                all_lidar_coords.append(new_coord)

            all_lidar_feat = torch.cat(all_lidar_feat, 0)
            all_lidar_coords = torch.cat(all_lidar_coords)

            input_sp_tensor = spconv.SparseConvTensor(
                features=all_lidar_feat,
                indices=all_lidar_coords.int(),
                spatial_shape=new_shape,
                batch_size=batch_size
            )
            x = self.conv_input(input_sp_tensor)

            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)

            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)

            for i in range(stages):
                if i==0:
                    stage_id = ''
                else:
                    stage_id = str(i)

                this_conv3 = self.decompose_tensor(x_conv3, i, batch_size)
                this_conv4 = self.decompose_tensor(x_conv4, i, batch_size)
                this_out = self.decompose_tensor(out, i, batch_size)

                batch_dict.update({
                    'encoded_spconv_tensor'+stage_id: this_out,
                    'encoded_spconv_tensor_stride'+stage_id: 8,
                })
                batch_dict.update({
                    'multi_scale_3d_features'+stage_id: {
                        'x_conv1': None,
                        'x_conv2': None,
                        'x_conv3': this_conv3,
                        'x_conv4': this_conv4,
                    },
                    'multi_scale_3d_strides'+stage_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })

        for i in range(stages):
            if i == 0:
                stage_id = ''
            else:
                stage_id = str(i)
            if self.model_cfg.get('MM', False):
                newvoxel_features, newvoxel_coords = batch_dict['voxel_features_mm' + stage_id], batch_dict[
                    'voxel_coords_mm' + stage_id]

                newinput_sp_tensor = spconv.SparseConvTensor(
                    features=newvoxel_features,
                    indices=newvoxel_coords.int(),
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_size
                )
                if self.training:
                    voxel_sampling(newinput_sp_tensor, 0.4)

                calib = batch_dict['calib']
                batch_size = batch_dict['batch_size']
                if 'aug_param' in batch_dict:
                    trans_param = batch_dict['aug_param']
                else:
                    trans_param = None
                if 'transform_param' in batch_dict:
                    trans_param = batch_dict['transform_param'][:, i, :]

                newx_conv1 = self.vir_conv1(newinput_sp_tensor, batch_size, calib, 1, self.x_trans_train, trans_param)

                if self.training:
                    voxel_sampling(newx_conv1, 0.2)

                newx_conv2 = self.vir_conv2(newx_conv1, batch_size, calib, 2, self.x_trans_train, trans_param)

                if self.training:
                    voxel_sampling(newx_conv2, 0.1)

                newx_conv3 = self.vir_conv3(newx_conv2, batch_size, calib, 4, self.x_trans_train, trans_param)

                if self.training:
                    voxel_sampling(newx_conv3, 0.05)

                newx_conv4 = self.vir_conv4(newx_conv3, batch_size, calib, 8, self.x_trans_train, trans_param)

                batch_dict.update({
                    'encoded_spconv_tensor_stride_mm' + stage_id: 8
                })
                batch_dict.update({
                    'multi_scale_3d_features_mm' + stage_id: {
                        'x_conv1': newx_conv1,
                        'x_conv2': newx_conv2,
                        'x_conv3': newx_conv3,
                        'x_conv4': newx_conv4,
                    },
                    'multi_scale_3d_strides' + stage_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })

        return batch_dict

class VirConv8xO(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, num_frames, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_frames = num_frames
        self.x_trans_train = X_TRAIN()

        self.return_num_features_as_dict = model_cfg.RETURN_NUM_FEATURES_AS_DICT
        self.out_features=model_cfg.OUT_FEATURES

        num_filters = model_cfg.NUM_FILTERS

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.vir_conv1 = VirConvBlock(input_channels, num_filters[0], stride=1, indice_key='vir1')
        self.vir_conv2 = VirConvBlock(num_filters[0], num_filters[1], stride=2, indice_key='vir2')
        self.vir_conv3 = VirConvBlock(num_filters[1], num_filters[2], stride=2, indice_key='vir3')
        self.vir_conv4 = VirConvBlock(num_filters[2], num_filters[3], stride=2, padding=(0, 1, 1), indice_key='vir4')

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(num_filters[3], self.out_features, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_features),
            nn.ReLU(),
        )

        self.num_point_features = self.out_features

        if self.return_num_features_as_dict:
            num_point_features = {}
            num_point_features.update({
                'x_conv1': num_filters[0],
                'x_conv2': num_filters[1],
                'x_conv3': num_filters[2],
                'x_conv4': num_filters[3],
            })
            self.num_point_features = num_point_features
        #for child in self.children():
        #    for param in child.parameters():
        #        param.requires_grad = False

    def decompose_tensor(self, tensor, i, batch_size):
        input_shape = tensor.spatial_shape[2]
        begin_shape_ids = i * (input_shape // 4)
        end_shape_ids = (i + 1) * (input_shape // 4)
        x_conv3_features = tensor.features
        x_conv3_coords = tensor.indices

        mask = (begin_shape_ids < x_conv3_coords[:, 3]) & (x_conv3_coords[:, 3] < end_shape_ids)
        this_conv3_feat = x_conv3_features[mask]
        this_conv3_coords = x_conv3_coords[mask]
        this_conv3_coords[:, 3] -= i * (input_shape // 4)
        this_shape = [tensor.spatial_shape[0], tensor.spatial_shape[1], tensor.spatial_shape[2] // 4]

        this_conv3_tensor = spconv.SparseConvTensor(
            features=this_conv3_feat,
            indices=this_conv3_coords.int(),
            spatial_shape=this_shape,
            batch_size=batch_size
        )
        return this_conv3_tensor

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            stages = trans_param.shape[1]
        else:
            stages = 1


        batch_size = batch_dict['batch_size']


        for i in range(stages):
            if i == 0:
                stage_id = ''
            else:
                stage_id = str(i)
            newvoxel_features, newvoxel_coords = batch_dict['voxel_features' + stage_id], batch_dict[
                'voxel_coords' + stage_id]

            newinput_sp_tensor = spconv.SparseConvTensor(
                features=newvoxel_features,
                indices=newvoxel_coords.int(),
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )
            if self.training:
                voxel_sampling(newinput_sp_tensor, 0.4)

            calib = batch_dict['calib']
            batch_size = batch_dict['batch_size']
            if 'aug_param' in batch_dict:
                trans_param = batch_dict['aug_param']
            else:
                trans_param = None
            if 'transform_param' in batch_dict:
                trans_param = batch_dict['transform_param'][:, i, :]

            newx_conv1 = self.vir_conv1(newinput_sp_tensor, batch_size, calib, 1, self.x_trans_train, trans_param)

            if self.training:
                voxel_sampling(newx_conv1, 0.2)

            newx_conv2 = self.vir_conv2(newx_conv1, batch_size, calib, 2, self.x_trans_train, trans_param)

            if self.training:
                voxel_sampling(newx_conv2, 0.1)

            newx_conv3 = self.vir_conv3(newx_conv2, batch_size, calib, 4, self.x_trans_train, trans_param)

            if self.training:
                voxel_sampling(newx_conv3, 0.05)

            newx_conv4 = self.vir_conv4(newx_conv3, batch_size, calib, 8, self.x_trans_train, trans_param)

            out = self.conv_out(newx_conv4)

            batch_dict.update({
                'encoded_spconv_tensor' + stage_id: out,
                'encoded_spconv_tensor_stride' + stage_id: 8,
            })

            batch_dict.update({
                'encoded_spconv_tensor_stride' + stage_id: 8
            })
            batch_dict.update({
                'multi_scale_3d_features' + stage_id: {
                    'x_conv1': newx_conv1,
                    'x_conv2': newx_conv2,
                    'x_conv3': newx_conv3,
                    'x_conv4': newx_conv4,
                },
                'multi_scale_3d_strides' + stage_id: {
                    'x_conv1': 1,
                    'x_conv2': 2,
                    'x_conv3': 4,
                    'x_conv4': 8,
                }
            })

        return batch_dict

class VirConvRes8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, num_frames, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_frames = num_frames
        self.x_trans_train = X_TRAIN()

        self.return_num_features_as_dict = model_cfg.RETURN_NUM_FEATURES_AS_DICT
        self.out_features=model_cfg.OUT_FEATURES

        num_filters = model_cfg.NUM_FILTERS

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(num_filters[0]),
            nn.ReLU(inplace=True),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(num_filters[0], num_filters[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(num_filters[0], num_filters[0], norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(num_filters[0], num_filters[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(num_filters[1], num_filters[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(num_filters[1], num_filters[1], norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(num_filters[1], num_filters[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(num_filters[2], num_filters[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(num_filters[2], num_filters[2], norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(num_filters[2], num_filters[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(num_filters[3], num_filters[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(num_filters[3], num_filters[3], norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(num_filters[3], self.out_features, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_features),
            nn.ReLU(),
        )
        if self.model_cfg.get('MM', False):
            self.vir_conv1 = VirConvBlock(input_channels, num_filters[0], stride=1, indice_key='vir1')
            self.vir_conv2 = VirConvBlock(num_filters[0], num_filters[1], stride=2, indice_key='vir2')
            self.vir_conv3 = VirConvBlock(num_filters[1], num_filters[2], stride=2, indice_key='vir3')
            self.vir_conv4 = VirConvBlock(num_filters[2], num_filters[3], stride=2, padding=(0,1,1), indice_key='vir4')

        self.num_point_features = self.out_features

        if self.return_num_features_as_dict:
            num_point_features = {}
            num_point_features.update({
                'x_conv1': num_filters[0],
                'x_conv2': num_filters[1],
                'x_conv3': num_filters[2],
                'x_conv4': num_filters[3],
            })
            self.num_point_features = num_point_features
        #for child in self.children():
        #    for param in child.parameters():
        #        param.requires_grad = False

    def decompose_tensor(self, tensor, i, batch_size):
        input_shape = tensor.spatial_shape[2]
        begin_shape_ids = i * (input_shape // 4)
        end_shape_ids = (i + 1) * (input_shape // 4)
        x_conv3_features = tensor.features
        x_conv3_coords = tensor.indices

        mask = (begin_shape_ids < x_conv3_coords[:, 3]) & (x_conv3_coords[:, 3] < end_shape_ids)
        this_conv3_feat = x_conv3_features[mask]
        this_conv3_coords = x_conv3_coords[mask]
        this_conv3_coords[:, 3] -= i * (input_shape // 4)
        this_shape = [tensor.spatial_shape[0], tensor.spatial_shape[1], tensor.spatial_shape[2] // 4]

        this_conv3_tensor = spconv.SparseConvTensor(
            features=this_conv3_feat,
            indices=this_conv3_coords.int(),
            spatial_shape=this_shape,
            batch_size=batch_size
        )
        return this_conv3_tensor

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            stages = trans_param.shape[1]
        else:
            stages = 1


        batch_size = batch_dict['batch_size']
        all_lidar_feat = []
        all_lidar_coords = []

        new_shape = [self.sparse_shape[0], self.sparse_shape[1], self.sparse_shape[2] * 4]

        if self.training:
            for i in range(stages):
                if i == 0:
                    stage_id = ''
                else:
                    stage_id = str(i)

                voxel_features, voxel_coords = batch_dict['voxel_features' + stage_id], batch_dict[
                    'voxel_coords' + stage_id]

                batch_size = batch_dict['batch_size']
                input_sp_tensor = spconv.SparseConvTensor(
                    features=voxel_features,
                    indices=voxel_coords.int(),
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_size
                )
                x = self.conv_input(input_sp_tensor)

                x_conv1 = self.conv1(x)
                x_conv2 = self.conv2(x_conv1)
                x_conv3 = self.conv3(x_conv2)
                x_conv4 = self.conv4(x_conv3)

                # for detection head
                # [200, 176, 5] -> [200, 176, 2]
                out = self.conv_out(x_conv4)

                batch_dict.update({
                    'encoded_spconv_tensor' + stage_id: out,
                    'encoded_spconv_tensor_stride' + stage_id: 8,
                })
                batch_dict.update({
                    'multi_scale_3d_features' + stage_id: {
                        'x_conv1': x_conv1,
                        'x_conv2': x_conv2,
                        'x_conv3': x_conv3,
                        'x_conv4': x_conv4,
                    },
                    'multi_scale_3d_strides' + stage_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })
        else:
            for i in range(stages):
                if i==0:
                    stage_id = ''
                else:
                    stage_id = str(i)

                voxel_features, voxel_coords = batch_dict['voxel_features'+stage_id], batch_dict['voxel_coords'+stage_id]

                all_lidar_feat.append(voxel_features)
                new_coord = voxel_coords.clone()
                new_coord[:, 3] += i*self.sparse_shape[2]
                all_lidar_coords.append(new_coord)

            all_lidar_feat = torch.cat(all_lidar_feat, 0)
            all_lidar_coords = torch.cat(all_lidar_coords)

            input_sp_tensor = spconv.SparseConvTensor(
                features=all_lidar_feat,
                indices=all_lidar_coords.int(),
                spatial_shape=new_shape,
                batch_size=batch_size
            )
            x = self.conv_input(input_sp_tensor)

            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)

            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)

            for i in range(stages):
                if i==0:
                    stage_id = ''
                else:
                    stage_id = str(i)

                this_conv3 = self.decompose_tensor(x_conv3, i, batch_size)
                this_conv4 = self.decompose_tensor(x_conv4, i, batch_size)
                this_out = self.decompose_tensor(out, i, batch_size)

                batch_dict.update({
                    'encoded_spconv_tensor'+stage_id: this_out,
                    'encoded_spconv_tensor_stride'+stage_id: 8,
                })
                batch_dict.update({
                    'multi_scale_3d_features'+stage_id: {
                        'x_conv1': None,
                        'x_conv2': None,
                        'x_conv3': this_conv3,
                        'x_conv4': this_conv4,
                    },
                    'multi_scale_3d_strides'+stage_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })

        for i in range(stages):
            if i == 0:
                stage_id = ''
            else:
                stage_id = str(i)
            if self.model_cfg.get('MM', False):
                newvoxel_features, newvoxel_coords = batch_dict['voxel_features_mm' + stage_id], batch_dict[
                    'voxel_coords_mm' + stage_id]

                newinput_sp_tensor = spconv.SparseConvTensor(
                    features=newvoxel_features,
                    indices=newvoxel_coords.int(),
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_size
                )
                if self.training:
                    voxel_sampling(newinput_sp_tensor, 0.4)

                calib = batch_dict['calib']
                batch_size = batch_dict['batch_size']
                if 'aug_param' in batch_dict:
                    trans_param = batch_dict['aug_param']
                else:
                    trans_param = None
                if 'transform_param' in batch_dict:
                    trans_param = batch_dict['transform_param'][:, i, :]

                newx_conv1 = self.vir_conv1(newinput_sp_tensor, batch_size, calib, 1, self.x_trans_train, trans_param)

                if self.training:
                    voxel_sampling(newx_conv1, 0.2)

                newx_conv2 = self.vir_conv2(newx_conv1, batch_size, calib, 2, self.x_trans_train, trans_param)

                if self.training:
                    voxel_sampling(newx_conv2, 0.1)

                newx_conv3 = self.vir_conv3(newx_conv2, batch_size, calib, 4, self.x_trans_train, trans_param)

                if self.training:
                    voxel_sampling(newx_conv3, 0.05)

                newx_conv4 = self.vir_conv4(newx_conv3, batch_size, calib, 8, self.x_trans_train, trans_param)

                batch_dict.update({
                    'encoded_spconv_tensor_stride_mm' + stage_id: 8
                })
                batch_dict.update({
                    'multi_scale_3d_features_mm' + stage_id: {
                        'x_conv1': newx_conv1,
                        'x_conv2': newx_conv2,
                        'x_conv3': newx_conv3,
                        'x_conv4': newx_conv4,
                    },
                    'multi_scale_3d_strides' + stage_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })

        return batch_dict

class VirConv2x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, num_frames, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_frames = num_frames
        self.x_trans_train = X_TRAIN()

        self.return_num_features_as_dict = model_cfg.RETURN_NUM_FEATURES_AS_DICT
        self.out_features=model_cfg.OUT_FEATURES

        num_filters = model_cfg.NUM_FILTERS

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(num_filters[0]),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(num_filters[0], num_filters[0], 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(num_filters[0], num_filters[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(num_filters[1], num_filters[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(num_filters[2], num_filters[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(num_filters[3], self.out_features, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_features),
            nn.ReLU(),
        )
        if self.model_cfg.get('MM', False):
            self.conv_input_2 = spconv.SparseSequential(
                spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1_2'),
                norm_fn(num_filters[0]),
                nn.ReLU(),
            )
            block = post_act_block

            self.conv1_2 = spconv.SparseSequential(
                block(num_filters[0], num_filters[0], 3, norm_fn=norm_fn, padding=1, indice_key='subm1_2'),
            )
            self.conv1_2_c = spconv.SparseSequential(
                block(input_channels, num_filters[0], (1, 3, 3), norm_fn=norm_fn, padding=(0, 1, 1),indice_key='subm1_2_c'),
                block(num_filters[0], num_filters[0], (1,3,3), norm_fn=norm_fn, padding=(0,1,1), indice_key='subm1_2_c'),
            )

            self.conv2_2 = spconv.SparseSequential(
                # [1600, 1408, 41] <- [800, 704, 21]
                block(num_filters[0]*2, num_filters[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2_2', conv_type='spconv'),
                block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2_2'),
            )

            self.conv2_2_c = spconv.SparseSequential(
                # [1600, 1408, 41] <- [800, 704, 21]
                block(num_filters[1], num_filters[1], (1,3,3), norm_fn=norm_fn, padding=(0,1,1), indice_key='subm2_2_c'),
                block(num_filters[1], num_filters[1], (1,3,3), norm_fn=norm_fn, padding=(0,1,1), indice_key='subm2_2_c'),
            )



        self.num_point_features = self.out_features

        if self.return_num_features_as_dict:
            num_point_features = {}
            num_point_features.update({
                'x_conv1': num_filters[0],
                'x_conv2': num_filters[1],
                'x_conv3': num_filters[2],
                'x_conv4': num_filters[3],
            })
            self.num_point_features = num_point_features
        #for child in self.children():
        #    for param in child.parameters():
        #        param.requires_grad = False


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            stages = trans_param.shape[1]
        else:
            stages = 1


        for i in range(stages):
            if i==0:
                stage_id = ''
            else:
                stage_id = str(i)

            voxel_features, voxel_coords = batch_dict['voxel_features'+stage_id], batch_dict['voxel_coords'+stage_id]

            batch_size = batch_dict['batch_size']
            input_sp_tensor = spconv.SparseConvTensor(
                features=voxel_features,
                indices=voxel_coords.int(),
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )
            x = self.conv_input(input_sp_tensor)

            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)

            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)

            batch_dict.update({
                'encoded_spconv_tensor'+stage_id: out,
                'encoded_spconv_tensor_stride'+stage_id: 8,
            })
            batch_dict.update({
                'multi_scale_3d_features'+stage_id: {
                    'x_conv1': x_conv1,
                    'x_conv2': x_conv2,
                    'x_conv3': x_conv3,
                    'x_conv4': x_conv4,
                },
                'multi_scale_3d_strides'+stage_id: {
                    'x_conv1': 1,
                    'x_conv2': 2,
                    'x_conv3': 4,
                    'x_conv4': 8,
                }
            })

            if self.model_cfg.get('MM', False):
                newvoxel_features, newvoxel_coords = batch_dict['voxel_features_mm'+stage_id], batch_dict['voxel_coords_mm'+stage_id]
                #point_coords = newvoxel_features[:, 0:3]
                #point_depths = point_coords.norm(dim=1) / 70 - 0.5
                #newvoxel_features[:, -1] = point_depths

                calib = batch_dict['calib']
                batch_size = batch_dict['batch_size']
                if 'aug_param' in batch_dict:
                    trans_param = batch_dict['aug_param']
                else:
                    trans_param = None
                if 'transform_param' in batch_dict:
                    trans_param = batch_dict['transform_param'][:,  i, :]
                uv_coords = index2uv(newvoxel_coords, batch_size, calib, 1, self.x_trans_train, trans_param)

                newinput_sp_tensor = spconv.SparseConvTensor(
                    features=newvoxel_features,
                    indices=newvoxel_coords.int(),
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_size
                )

                new_uv_tensor = spconv.SparseConvTensor(
                    features=newvoxel_features,
                    indices=uv_coords.int(),
                    spatial_shape=[10, 1600, 600],
                    batch_size=batch_size
                )

                newx = self.conv_input_2(newinput_sp_tensor)

                newx_conv1 = self.conv1_2(newx)
                new_uv_conv1 = self.conv1_2_c(new_uv_tensor)
                newx_conv1 = replace_feature(newx_conv1, torch.cat([newx_conv1.features, new_uv_conv1.features],-1))

                newx_conv2 = self.conv2_2(newx_conv1)


                uv_coords = index2uv(newx_conv2.indices, batch_size, calib, 2, self.x_trans_train, trans_param)
                new_uv_conv2_tensor = spconv.SparseConvTensor(
                    features=newx_conv2.features,
                    indices=uv_coords.int(),
                    spatial_shape=[10, 1600, 600],
                    batch_size=batch_size
                )
                new_uv_conv2 = self.conv2_2_c(new_uv_conv2_tensor)
                newx_conv2=replace_feature(newx_conv2, torch.cat([newx_conv2.features, new_uv_conv2.features],-1))

                # for detection head
                # [200, 176, 5] -> [200, 176, 2]
                #newout = self.conv_out(newx_conv4)

                batch_dict.update({
                    #'encoded_spconv_tensor_mm': newout,
                    'encoded_spconv_tensor_stride_mm'+stage_id: 8
                })
                batch_dict.update({
                    'multi_scale_3d_features_mm'+stage_id: {
                        'x_conv1': newx_conv1,
                        'x_conv2': newx_conv2,
                    },
                    'multi_scale_3d_strides'+stage_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })

        return batch_dict

class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, num_frames, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_frames = num_frames

        self.return_num_features_as_dict = model_cfg.RETURN_NUM_FEATURES_AS_DICT
        self.out_features=model_cfg.OUT_FEATURES

        num_filters = model_cfg.NUM_FILTERS

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(num_filters[0]),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(num_filters[0], num_filters[0], 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(num_filters[0], num_filters[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(num_filters[1], num_filters[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(num_filters[2], num_filters[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(num_filters[3], self.out_features, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_features),
            nn.ReLU(),
        )
        if self.model_cfg.get('MM', False):
            self.conv_input_2 = spconv.SparseSequential(
                spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1_2'),
                norm_fn(num_filters[0]),
                nn.ReLU(),
            )
            block = post_act_block

            self.conv1_2 = spconv.SparseSequential(
                block(num_filters[0], num_filters[0], 3, norm_fn=norm_fn, padding=1, indice_key='subm1_2'),
            )

            self.conv2_2 = spconv.SparseSequential(
                # [1600, 1408, 41] <- [800, 704, 21]
                block(num_filters[0], num_filters[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2_2', conv_type='spconv'),
                block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2_2'),
                block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2_2'),
            )

            self.conv3_2 = spconv.SparseSequential(
                # [800, 704, 21] <- [400, 352, 11]
                block(num_filters[1], num_filters[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3_2', conv_type='spconv'),
                block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3_2'),
                block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3_2'),
            )

            self.conv4_2 = spconv.SparseSequential(
                # [400, 352, 11] <- [200, 176, 5]
                block(num_filters[2], num_filters[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4_2', conv_type='spconv'),
                block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4_2'),
                block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4_2'),
            )

        self.num_point_features = self.out_features

        if self.return_num_features_as_dict:
            num_point_features = {}
            num_point_features.update({
                'x_conv1': num_filters[0],
                'x_conv2': num_filters[1],
                'x_conv3': num_filters[2],
                'x_conv4': num_filters[3],
            })
            self.num_point_features = num_point_features
        #for child in self.children():
        #    for param in child.parameters():
        #        param.requires_grad = False
    def decompose_tensor(self, tensor, i, batch_size):
        input_shape = tensor.spatial_shape[2]
        begin_shape_ids = i * (input_shape // 4)
        end_shape_ids = (i + 1) * (input_shape // 4)
        x_conv3_features = tensor.features
        x_conv3_coords = tensor.indices

        mask = (begin_shape_ids < x_conv3_coords[:, 3]) & (x_conv3_coords[:, 3] < end_shape_ids)
        this_conv3_feat = x_conv3_features[mask]
        this_conv3_coords = x_conv3_coords[mask]
        this_conv3_coords[:, 3] -= i * (input_shape // 4)
        this_shape = [tensor.spatial_shape[0], tensor.spatial_shape[1], tensor.spatial_shape[2] // 4]

        this_conv3_tensor = spconv.SparseConvTensor(
            features=this_conv3_feat,
            indices=this_conv3_coords.int(),
            spatial_shape=this_shape,
            batch_size=batch_size
        )
        return this_conv3_tensor

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            stages = trans_param.shape[1]
        else:
            stages = 1


        batch_size = batch_dict['batch_size']
        all_lidar_feat = []
        all_lidar_coords = []

        new_shape = [self.sparse_shape[0], self.sparse_shape[1], self.sparse_shape[2] * 4]

        if self.training:
            for i in range(stages):
                if i == 0:
                    stage_id = ''
                else:
                    stage_id = str(i)

                voxel_features, voxel_coords = batch_dict['voxel_features' + stage_id], batch_dict[
                    'voxel_coords' + stage_id]

                batch_size = batch_dict['batch_size']
                input_sp_tensor = spconv.SparseConvTensor(
                    features=voxel_features,
                    indices=voxel_coords.int(),
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_size
                )
                x = self.conv_input(input_sp_tensor)

                x_conv1 = self.conv1(x)
                x_conv2 = self.conv2(x_conv1)
                x_conv3 = self.conv3(x_conv2)
                x_conv4 = self.conv4(x_conv3)

                # for detection head
                # [200, 176, 5] -> [200, 176, 2]
                out = self.conv_out(x_conv4)

                batch_dict.update({
                    'encoded_spconv_tensor' + stage_id: out,
                    'encoded_spconv_tensor_stride' + stage_id: 8,
                })
                batch_dict.update({
                    'multi_scale_3d_features' + stage_id: {
                        'x_conv1': x_conv1,
                        'x_conv2': x_conv2,
                        'x_conv3': x_conv3,
                        'x_conv4': x_conv4,
                    },
                    'multi_scale_3d_strides' + stage_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })
        else:
            for i in range(stages):
                if i==0:
                    stage_id = ''
                else:
                    stage_id = str(i)

                voxel_features, voxel_coords = batch_dict['voxel_features'+stage_id], batch_dict['voxel_coords'+stage_id]

                all_lidar_feat.append(voxel_features)
                new_coord = voxel_coords.clone()
                new_coord[:, 3] += i*self.sparse_shape[2]
                all_lidar_coords.append(new_coord)

            all_lidar_feat = torch.cat(all_lidar_feat, 0)
            all_lidar_coords = torch.cat(all_lidar_coords)

            input_sp_tensor = spconv.SparseConvTensor(
                features=all_lidar_feat,
                indices=all_lidar_coords.int(),
                spatial_shape=new_shape,
                batch_size=batch_size
            )
            x = self.conv_input(input_sp_tensor)

            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)

            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)

            for i in range(stages):
                if i==0:
                    stage_id = ''
                else:
                    stage_id = str(i)

                this_conv3 = self.decompose_tensor(x_conv3, i, batch_size)
                this_conv4 = self.decompose_tensor(x_conv4, i, batch_size)
                this_out = self.decompose_tensor(out, i, batch_size)

                batch_dict.update({
                    'encoded_spconv_tensor'+stage_id: this_out,
                    'encoded_spconv_tensor_stride'+stage_id: 8,
                })
                batch_dict.update({
                    'multi_scale_3d_features'+stage_id: {
                        'x_conv1': None,
                        'x_conv2': None,
                        'x_conv3': this_conv3,
                        'x_conv4': this_conv4,
                    },
                    'multi_scale_3d_strides'+stage_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })

        return batch_dict

class SimpleVoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, num_frames,**kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_frames=num_frames
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(input_channels, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 64, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(64),
            nn.ReLU(),
        )
        self.num_point_features = 64

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        for i in range(1,self.num_frames):
            if 'voxel_features'+str(-i) not in batch_dict.keys():
                continue
            voxel_features, voxel_coords = batch_dict['voxel_features'+str(-i)], batch_dict['voxel_coords'+str(-i)]
            batch_size = batch_dict['batch_size']
            input_sp_tensor = spconv.SparseConvTensor(
                features=voxel_features,
                indices=voxel_coords.int(),
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )

            x_conv1 = self.conv1(input_sp_tensor)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)

            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)

            batch_dict.update({
                'encoded_spconv_tensor'+str(-i): out})


        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, num_frames, **kwargs):
        super().__init__()

        self.num_frames = num_frames

        self.return_num_features_as_dict = model_cfg.RETURN_NUM_FEATURES_AS_DICT
        self.out_features = model_cfg.OUT_FEATURES

        num_filters = model_cfg.NUM_FILTERS

        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(num_filters[0]),
            nn.ReLU(inplace=True),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(num_filters[0], num_filters[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(num_filters[0], num_filters[0], norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(num_filters[0], num_filters[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(num_filters[1], num_filters[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(num_filters[1], num_filters[1], norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(num_filters[1], num_filters[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(num_filters[2], num_filters[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(num_filters[2], num_filters[2], norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(num_filters[2], num_filters[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(num_filters[3], num_filters[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(num_filters[3], num_filters[3], norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(num_filters[3], self.out_features, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_features),
            nn.ReLU(),
        )

        if self.model_cfg.get('MM', False):

            self.conv_input_2 = spconv.SparseSequential(
                spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1_2'),
                norm_fn(num_filters[0]),
                nn.ReLU(inplace=True),
            )
            block = post_act_block

            self.conv1_2 = spconv.SparseSequential(
                SparseBasicBlock(num_filters[0], num_filters[0], norm_fn=norm_fn, indice_key='res1_2'),
                SparseBasicBlock(num_filters[0], num_filters[0], norm_fn=norm_fn, indice_key='res1_2'),
            )

            self.conv2_2 = spconv.SparseSequential(
                # [1600, 1408, 41] <- [800, 704, 21]
                block(num_filters[0], num_filters[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2_2', conv_type='spconv'),
                SparseBasicBlock(num_filters[1], num_filters[1], norm_fn=norm_fn, indice_key='res2_2'),
            )

            self.conv3_2 = spconv.SparseSequential(
                # [800, 704, 21] <- [400, 352, 11]
                block(num_filters[1], num_filters[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3_2', conv_type='spconv'),
                SparseBasicBlock(num_filters[2], num_filters[2], norm_fn=norm_fn, indice_key='res3_2'),
            )

            self.conv4_2 = spconv.SparseSequential(
                # [400, 352, 11] <- [200, 176, 5]
                block(num_filters[2], num_filters[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4_2', conv_type='spconv'),
                SparseBasicBlock(num_filters[3], num_filters[3], norm_fn=norm_fn, indice_key='res4_2'),
            )


        self.num_point_features = self.out_features

        if self.return_num_features_as_dict:
            num_point_features = {}
            num_point_features.update({
                'x_conv1': num_filters[0],
                'x_conv2': num_filters[1],
                'x_conv3': num_filters[2],
                'x_conv4': num_filters[3],
            })
            self.num_point_features = num_point_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            stages = trans_param.shape[1]
        else:
            stages = 1



        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']

        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            },
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        if self.training:

            newvoxel_features, newvoxel_coords = batch_dict['voxel_features1'], batch_dict['voxel_coords1']

            newinput_sp_tensor = spconv.SparseConvTensor(
                features=newvoxel_features,
                indices=newvoxel_coords.int(),
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )
            newx = self.conv_input_2(newinput_sp_tensor)

            newx_conv1 = self.conv1_2(newx)
            newx_conv2 = self.conv2_2(newx_conv1)
            newx_conv3 = self.conv3_2(newx_conv2)
            newx_conv4 = self.conv4_2(newx_conv3)

            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            #newout = self.conv_out(newx_conv4)

            batch_dict.update({
                #'encoded_spconv_tensor_mm': newout,
                'encoded_spconv_tensor_stride_mm': 8
            })
            batch_dict.update({
                'multi_scale_3d_features_mm': {
                    'x_conv1': newx_conv1,
                    'x_conv2': newx_conv2,
                    'x_conv3': newx_conv3,
                    'x_conv4': newx_conv4,
                },
                'multi_scale_3d_strides': {
                    'x_conv1': 1,
                    'x_conv2': 2,
                    'x_conv3': 4,
                    'x_conv4': 8,
                }
            })

        return batch_dict


class VoxelResBackBone4x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, num_frames,**kwargs):
        super().__init__()

        self.num_frames = num_frames

        self.return_num_features_as_dict = model_cfg.RETURN_NUM_FEATURES_AS_DICT
        self.out_features = model_cfg.OUT_FEATURES

        num_filters = model_cfg.NUM_FILTERS

        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(num_filters[0]),
            nn.ReLU(inplace=True),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(num_filters[0], num_filters[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(num_filters[0], num_filters[0], norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(num_filters[0], num_filters[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(num_filters[1], num_filters[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(num_filters[1], num_filters[1], norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(num_filters[1], num_filters[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(num_filters[2], num_filters[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(num_filters[2], num_filters[2], norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(num_filters[2], num_filters[3], 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(num_filters[3], num_filters[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(num_filters[3], num_filters[3], norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(num_filters[3], self.out_features, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_features),
            nn.ReLU(),
        )
        self.num_point_features = self.out_features

        if self.return_num_features_as_dict:
            num_point_features = {}
            num_point_features.update({
                'x_conv1': num_filters[0],
                'x_conv2': num_filters[1],
                'x_conv3': num_filters[2],
                'x_conv4': num_filters[3],
            })
            self.num_point_features = num_point_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']

        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            },
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class Voxel_MAE(nn.Module):
    """
    pre-trained model
    """

    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.masked_ratio = model_cfg.MASKED_RATIO

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
            last_pad = self.model_cfg.get('last_pad', 0)

            self.conv_out = spconv.SparseSequential(
                # [200, 150, 5] -> [200, 150, 2]
                spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                    bias=False, indice_key='spconv_down2'),
                norm_fn(128),
                nn.ReLU(),
            )
        else:
            self.conv_out = None

        self.num_point_features = 16

        if self.model_cfg.get('MASKED_RATIO', 0)>0:

            self.deconv1 = nn.Sequential(
                nn.ConvTranspose3d(128, 32, 3, padding=1, output_padding=1, stride=2, bias=False),
                nn.BatchNorm3d(32),
                nn.ReLU()
            )
            self.deconv2 = nn.Sequential(
                nn.ConvTranspose3d(32, 8, 3, padding=1, output_padding=1, stride=(4, 2, 2), bias=False),
                nn.BatchNorm3d(8),
                nn.ReLU()
            )
            self.deconv3 = nn.Sequential(
                nn.ConvTranspose3d(8, 1, 3, padding=1, output_padding=1, stride=(3, 2, 2), bias=False),
            )
            self.criterion = nn.BCEWithLogitsLoss()
            self.forward_re_dict = {}

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        pred = self.forward_re_dict['pred']
        target = self.forward_re_dict['target']
        loss = self.criterion(pred, target)

        tb_dict = {
            'loss_rpn': loss.item()
        }

        return loss, tb_dict

    def argwhere(self, tensor):

        new_t = tensor.cpu().numpy()
        new_t = np.argwhere(new_t)
        new_t = torch.from_numpy(new_t)
        new_t.to(tensor.device)

        return new_t

    def forward_mae(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """

        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']

        select_ratio = 1 - self.masked_ratio  # ratio for select voxel

        voxel_coords_distance = (voxel_coords[:, 2] ** 2 + voxel_coords[:, 3] ** 2) ** 0.5

        select_30 = voxel_coords_distance[:] <= 30
        select_30to50 = (voxel_coords_distance[:] > 30) & (voxel_coords_distance[:] <= 50)
        select_50 = voxel_coords_distance[:] > 50

        # id_list = [i for i in range(coords.shape[0])]
        id_list_select_30 = self.argwhere(select_30 == True).reshape(self.argwhere(select_30 == True).shape[0])
        id_list_select_30to50 = self.argwhere(select_30to50 == True).reshape(
            self.argwhere(select_30to50 == True).shape[0])
        id_list_select_50 = self.argwhere(select_50 == True).reshape(self.argwhere(select_50 == True).shape[0])

        shuffle_id_list_select_30 = id_list_select_30
        random.shuffle(shuffle_id_list_select_30)

        shuffle_id_list_select_30to50 = id_list_select_30to50
        random.shuffle(shuffle_id_list_select_30to50)

        shuffle_id_list_select_50 = id_list_select_50
        random.shuffle(shuffle_id_list_select_50)

        slect_index = torch.cat((shuffle_id_list_select_30[:int(select_ratio * len(shuffle_id_list_select_30))],
                                 shuffle_id_list_select_30to50[
                                 :int((select_ratio + 0.2) * len(shuffle_id_list_select_30to50))],
                                 shuffle_id_list_select_50[:int((select_ratio + 0.2) * len(shuffle_id_list_select_50))]
                                 ), 0)

        nums = voxel_features.shape[0]

        voxel_fratures_all_one = torch.ones(nums, 1).to(voxel_features.device)
        voxel_features_partial, voxel_coords_partial = voxel_features[slect_index, :], voxel_coords[slect_index, :]

        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features_partial,
            indices=voxel_coords_partial.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        input_sp_tensor_ones = spconv.SparseConvTensor(
            features=voxel_fratures_all_one,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)

        self.forward_re_dict['target'] = input_sp_tensor_ones.dense()
        x_up1 = self.deconv1(out.dense())
        x_up2 = self.deconv2(x_up1)
        x_up3 = self.deconv3(x_up2)

        self.forward_re_dict['pred'] = x_up3

        return batch_dict

    def forward(self, batch_dict):

        if self.model_cfg.get('MASKED_RATIO', 0) > 0:
            return self.forward_mae(batch_dict)
        else:
            voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']

            batch_size = batch_dict['batch_size']
            input_sp_tensor = spconv.SparseConvTensor(
                features=voxel_features,
                indices=voxel_coords.int(),
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )
            x = self.conv_input(input_sp_tensor)

            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)

            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)

            batch_dict.update({
                'encoded_spconv_tensor': out,
                'encoded_spconv_tensor_stride': 8
            })
            batch_dict.update({
                'multi_scale_3d_features': {
                    'x_conv1': x_conv1,
                    'x_conv2': x_conv2,
                    'x_conv3': x_conv3,
                    'x_conv4': x_conv4,
                },
                'multi_scale_3d_strides': {
                    'x_conv1': 1,
                    'x_conv2': 2,
                    'x_conv3': 4,
                    'x_conv4': 8,
                }
            })
            return batch_dict
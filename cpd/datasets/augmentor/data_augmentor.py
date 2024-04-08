from functools import partial

import numpy as np

from ...utils import common_utils
from . import augmentor_utils, database_sampler


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None, num_frames=1,dataset_cfg=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger
        self.num_frames = num_frames
        self.dataset_cfg = dataset_cfg

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger,
            num_frames=self.num_frames,
            dataset_cfg = self.dataset_cfg
        )
        return db_sampler

    def da_sampling(self, config=None):
        db_sampler = database_sampler.DADataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger,
            num_frames=self.num_frames
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
   


    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        if self.num_frames==1:
            gt_boxes, points, param = augmentor_utils.global_rotation(
                data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
            )

            data_dict['gt_boxes'] = gt_boxes
            data_dict['points'] = points
            aug_param=[param]
            data_dict['aug_param'] = aug_param

        else:
            noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
            data_dict=augmentor_utils.global_rotation_with_param(data_dict, noise_rotation, self.num_frames)

        return data_dict

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)

        if self.num_frames==1:
            gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
            for cur_axis in config['ALONG_AXIS_LIST']:
                assert cur_axis in ['x', 'y']
                gt_boxes, points, param = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                    gt_boxes, points,
                )

            data_dict['gt_boxes'] = gt_boxes
            data_dict['points'] = points
            if 'aug_param' in data_dict:
                data_dict['aug_param'].append(int(param))
            else:
                data_dict['aug_param'] = [param]
        else:
            enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
            gt_boxes= data_dict['gt_boxes']
            for cur_axis in config['ALONG_AXIS_LIST']:
                assert cur_axis in ['x']
                gt_boxes = getattr(augmentor_utils, 'random_flip_with_param')(
                    gt_boxes, enable, ax=1)
                gt_boxes = getattr(augmentor_utils, 'random_flip_with_param')(
                        gt_boxes, enable, ax=6)
                data_dict['gt_boxes'] = gt_boxes

                for i in range(self.num_frames):
                    if i==0:
                        points=data_dict['points']
                        points = getattr(augmentor_utils, 'random_flip_with_param')(
                            points, enable, ax=1)
                        data_dict['points']=points
                    else:
                        if 'points'+str(i) in data_dict:

                            points = data_dict['points'+str(i)]
                            points = getattr(augmentor_utils, 'random_flip_with_param')(
                                points, enable, ax=1)
                            data_dict['points' + str(i)]=points

                        if 'gt_boxes' + str(i) in data_dict:
                            data_dict['gt_boxes' + str(i)] = getattr(augmentor_utils, 'random_flip_with_param')(
                                data_dict['gt_boxes' + str(i)], enable, ax=1)
                            data_dict['gt_boxes' + str(i)] = getattr(augmentor_utils, 'random_flip_with_param')(
                                data_dict['gt_boxes' + str(i)], enable, ax=6)


        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        if self.num_frames==1:
            gt_boxes, points, param = augmentor_utils.global_scaling(
                data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE']
            )
            data_dict['gt_boxes'] = gt_boxes
            data_dict['points'] = points
            if 'aug_param' in data_dict:
                data_dict['aug_param'].append(param)
            else:
                data_dict['aug_param'] = [param]

            return data_dict
        else:
            scale_range=config['WORLD_SCALE_RANGE']
            noise_scale = np.random.uniform(scale_range[0], scale_range[1])
            data_dict =  augmentor_utils.global_scaling_with_param(data_dict,noise_scale,self.num_frames)
            return data_dict

    def box_cut(box, cloud_in, scale=1.0):
        """
        input:
            box: array, shape=(7,)  (x, y, z, l, w, h, yaw)
            cloud: array, shape(N,M), (x, y, z, intensity, ...)
            scale: float, factor to enlarge the box size
        output:
            pts_in: array, points in box
            pts_out: array, points outside box
        """

        cloud = np.zeros(shape=(cloud_in.shape[0], 4))
        cloud[:, 0:3] = cloud_in[:, 0:3]
        cloud[:, 3] = 1

        x, y, z, l, w, h, yaw = box[0], box[1], box[2], box[3], box[4], box[5], box[6]

        trans_mat = np.eye(4, dtype=np.float32)
        trans_mat[0, 0] = np.cos(yaw)
        trans_mat[0, 1] = -np.sin(yaw)
        trans_mat[0, 3] = x
        trans_mat[1, 0] = np.sin(yaw)
        trans_mat[1, 1] = np.cos(yaw)
        trans_mat[1, 3] = y
        trans_mat[2, 3] = z

        trans_mat_i = np.linalg.inv(trans_mat)
        cloud = np.matmul(cloud, trans_mat_i.T)

        mask_l = np.logical_and(cloud[:, 0] < l * scale / 2, cloud[:, 0] > -l * scale / 2)
        mask_w = np.logical_and(cloud[:, 1] < w * scale / 2, cloud[:, 1] > -w * scale / 2)
        mask_h = np.logical_and(cloud[:, 2] < h * scale / 2, cloud[:, 2] > -h * scale / 2)
        mask = np.logical_and(np.logical_and(mask_w, mask_l), mask_h)
        mask_not = np.logical_not(mask)
        pts_in = cloud_in[mask]
        pts_out = cloud_in[mask_not]

        return pts_in, pts_out

    def random_local_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_local_flip, config=config)
        gt_boxes = data_dict['gt_boxes']
        points = data_dict['points']
        valid_mask = data_dict['valid_noise']

        for i, box in enumerate(gt_boxes):
            if valid_mask[i]:
                cloud = np.zeros(shape=(points.shape[0], 4))
                cloud[:, 0:3] = points[:, 0:3]
                cloud[:, 3] = 1

                x, y, z, l, w, h, yaw = box[0], box[1], box[2], box[3], box[4], box[5], box[6]

                trans_mat = np.eye(4, dtype=np.float32)
                trans_mat[0, 0] = np.cos(yaw)
                trans_mat[0, 1] = -np.sin(yaw)
                trans_mat[0, 3] = x
                trans_mat[1, 0] = np.sin(yaw)
                trans_mat[1, 1] = np.cos(yaw)
                trans_mat[1, 3] = y
                trans_mat[2, 3] = z

                trans_mat_i = np.linalg.inv(trans_mat)
                cloud = np.matmul(cloud, trans_mat_i.T)
                mask_l = np.logical_and(cloud[:, 0] < l / 2, cloud[:, 0] > -l / 2)
                mask_w = np.logical_and(cloud[:, 1] < w / 2, cloud[:, 1] > -w / 2)
                mask_h = np.logical_and(cloud[:, 2] < h / 2, cloud[:, 2] > -h / 2)
                mask = np.logical_and(np.logical_and(mask_w, mask_l), mask_h)

                subset_previous = points[mask]
                subset_new = cloud[mask]
                if np.random.choice([0, 1]):
                    subset_new[:, 0] = -subset_new[:, 0]
                    gt_boxes[i, 6] = np.pi + gt_boxes[i, 6]
                if np.random.choice([0, 1]):
                    subset_new[:, 1] = -subset_new[:, 1]
                subset_new = np.matmul(subset_new, trans_mat.T)
                subset_previous[:,0:3] = subset_new[:,0:3]

                points[mask] = subset_previous


        return data_dict

    def random_patch_shift(self, points, min=0, max=70.4, axis = 0):

        mid = np.random.random()*(max-min)+min

        points[points[:,axis]<min]=-1000
        points[points[:,axis]>max]=1000

        points_min = points[points[:, axis] <= mid]
        points_max = points[points[:, axis] > mid]

        points_min[:, axis] += (max-mid)
        points_max[:, axis] -= (mid - min)

        return np.concatenate([points_min, points_max])


    def random_world_trans(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_trans, config=config)

        if np.random.choice([0, 1]):
            points = data_dict['points']
            data_dict['points'] = self.random_patch_shift(points, 0, 70.4, 0)
            boxes = data_dict['gt_boxes']
            data_dict['gt_boxes'] = self.random_patch_shift(boxes, 0, 70.4, 0)

        if np.random.choice([0, 1]):
            points = data_dict['points']
            data_dict['points'] = self.random_patch_shift(points, -40, 40, 1)
            boxes = data_dict['gt_boxes']
            data_dict['gt_boxes'] = self.random_patch_shift(boxes, -40, 40, 1)

        std_x = (np.random.random()-0.5)*config.TRANSLATION_STD[0]
        std_y = (np.random.random()-0.5)*config.TRANSLATION_STD[1]
        std_z = (np.random.random()-0.5)*config.TRANSLATION_STD[2]
        param = np.array([std_x,std_y,std_z])

        data_dict['gt_boxes'][:,0:3] += param
        data_dict['points'][:,0:3] += param

        return data_dict



    def random_local_noise(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_local_noise, config=config)
        data_dict['gt_boxes'][:, 6] = -data_dict['gt_boxes'][:, 6]
        augmentor_utils.noise_per_object_v3_(data_dict['gt_boxes'], data_dict['points'], None,
                                        data_dict.get('valid_noise', None),
                                        config['LOCAL_ROT_RANGE'], config['TRANSLATION_STD'],
                                        config['GLOBAL_ROT_RANGE'], config['EXTRA_WIDTH'])
        data_dict['gt_boxes'][:, 6] = -data_dict['gt_boxes'][:, 6]
        if 'valid_noise' in data_dict:
            data_dict.pop('valid_noise')
        return data_dict

    def random_local_pyramid_aug(self, data_dict=None, config=None):
        """
        Refer to the paper:
            SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud
        """
        if data_dict is None:
            return partial(self.random_local_pyramid_aug, config=config)

        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']

        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_dropout(gt_boxes, points, config['DROP_PROB'])
        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_sparsify(gt_boxes, points,
                                                                            config['SPARSIFY_PROB'],
                                                                            config['SPARSIFY_MAX_NUM'],
                                                                            pyramids)
        gt_boxes, points = augmentor_utils.local_pyramid_swap(gt_boxes, points,
                                                              config['SWAP_PROB'],
                                                              config['SWAP_MAX_NUM'],
                                                              pyramids)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...
        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'aug_param' in data_dict:
            data_dict['aug_param'] = np.array(data_dict['aug_param'])
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')


        return data_dict

# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '7'
import pickle
import copy
import numpy as np
import torch
import tqdm
from pathlib import Path
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, common_utils
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.unsupervised_core.precompute_ppscore import save_pp_score
from pcdet.unsupervised_core import compute_outline_box
import multiprocessing
from functools import partial
from pcdet.unsupervised_core.ob_op import box_cut,la_sampling
import time
from pcdet.datasets.waymo_unsupervised.merge import DetectionsMerger


class WaymoUnsupervisedDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, test_iter=0):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger,
            test_iter=test_iter
        )
        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]

        self.only_top_lidar = dataset_cfg.TL
        self.sampling = dataset_cfg.SAMPLING

        self.infos = []
        self.include_waymo_data(self.mode)


    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []
        self.include_waymo_data(self.mode)

    def include_waymo_data(self, mode):
        self.logger.info('Loading Waymo dataset')
        waymo_infos = []

        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]

            outline_info_name = self.dataset_cfg.InitLabelGenerator

            if 'LabelRefiner' in self.dataset_cfg:
                outline_info_name = self.dataset_cfg.LabelRefiner

            if os.path.exists(self.data_path / sequence_name / (str(sequence_name)+'_outline_'+str(outline_info_name)+'.pkl')):
                info_path = self.data_path / sequence_name / (str(sequence_name)+'_outline_'+str(outline_info_name)+'.pkl')
            else:
                info_path = self.data_path / sequence_name / ('%s.pkl' % sequence_name)
            info_path = self.check_sequence_name_with_all_version(info_path)
            if not info_path.exists():
                num_skipped_infos += 1
                continue
            
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                waymo_infos.extend(infos)
            
        self.infos.extend(waymo_infos[:])
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos)))

        self.all_infos = self.infos

        self.index_list = list(range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]))

        if self.dataset_cfg.SAMPLED_INTERVAL[mode] > 1:
            sampled_waymo_infos = []
            for k in self.index_list:
                sampled_waymo_infos.append(self.infos[k])
            self.infos = sampled_waymo_infos

            self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.infos))


    @staticmethod
    def check_sequence_name_with_all_version(sequence_file):
        if '_with_camera_labels' not in str(sequence_file) and not sequence_file.exists():
            sequence_file = Path(str(sequence_file[:-9]) + '_with_camera_labels.tfrecord')
        if '_with_camera_labels' in str(sequence_file) and not sequence_file.exists():
            sequence_file = Path(str(sequence_file).replace('_with_camera_labels', ''))

        return sequence_file

    def get_infos(self, raw_data_path, save_path, num_workers=multiprocessing.cpu_count(), has_label=True, sampled_interval=1):
        from functools import partial
        # import sys
        # sys.path.append("./")
        import waymo_utils
        # from . import waymo_utils
        print('---------------The waymo sample interval is %d, total sequecnes is %d-----------------'
              % (sampled_interval, len(self.sample_sequence_list)))

        process_single_sequence = partial(
            waymo_utils.process_single_sequence,
            save_path=save_path, sampled_interval=sampled_interval, has_label=has_label
        )
        sample_sequence_file_list = [
            self.check_sequence_name_with_all_version(raw_data_path / sequence_file)
            for sequence_file in self.sample_sequence_list
        ]

        # process_single_sequence(sample_sequence_file_list[0])
        #with futures.ThreadPoolExecutor(num_workers) as executor:
        #    sequence_infos = list(tqdm(executor.map(process_single_sequence, sample_sequence_file_list),
        #                               total=len(sample_sequence_file_list)))
        all_sequences_infos=[]
        for i in tqdm.trange(len(sample_sequence_file_list)):
            single_sequence_file=sample_sequence_file_list[i]
            this_sequence_infos=process_single_sequence(single_sequence_file)
            all_sequences_infos+=this_sequence_infos
        #all_sequences_infos = [item for infos in sequence_infos for item in infos]

        return all_sequences_infos

    def get_lidar(self, sequence_name, sample_idx):
        lidar_file = self.data_path / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file).astype(np.float32)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]

        points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
        points_all = points_all[NLZ_flag == -1]
        points_all[:, 3] = np.tanh(points_all[:, 3])
        return points_all

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def compute_proto_pos(self, boxes):

        this_box = np.array(boxes)

        random_pose = (np.random.random(size=(2,)) - 0.5) * 120

        dis = np.linalg.norm(this_box[:, 0:2] - random_pose, axis=-1)
        if np.min(dis) > 4:
            return random_pose
        else:
            return self.compute_proto_pos(boxes)


    def radius_sampling(self, points, dis=[5, 5, 5, 5], intev=[7, 5, 4, 2]):
        distance = np.sqrt(np.sum(points[:, 0:2] ** 2, 1))

        points_list = []

        dis_iter = 0

        for i in range(len(dis)):
            dis_thresh = dis[i]
            sample_interval = intev[i]

            pos1 = dis_iter < distance
            pos2 = distance <= dis_iter + dis_thresh

            this_points = points[pos1 * pos2]
            sampling_flag = np.arange(0, this_points.shape[0])
            sampling_flag = sampling_flag % sample_interval == 0
            sampling_flag = sampling_flag.astype(bool)
            this_points = this_points[sampling_flag]
            points_list.append(this_points)
            dis_iter += dis_thresh

        points_list.append(points[distance > dis_iter])

        return np.concatenate(points_list)


    def points_rigid_transform(self, cloud, pose):
        if cloud.shape[0] == 0:
            return cloud
        mat = np.ones(shape=(cloud.shape[0], 4), dtype=np.float32)
        pose_mat = np.mat(pose)
        mat[:, 0:3] = cloud[:, 0:3]
        mat = np.mat(mat)
        transformed_mat = pose_mat * mat.T
        T = np.array(transformed_mat.T, dtype=np.float32)
        return T[:, 0:3]


    def sample_prototype(self, seq_name,
                       root_path,
                       points,
                       outline_boxes,
                       outline_cls,
                       outline_score,
                       proto_id,
                       dataset_cfg):

        new_outline_boxes = []
        new_outline_cls = []
        new_outline_score = []
        new_proto_id = []

        init_method_name = dataset_cfg.InitLabelGenerator
        proto_info_path = os.path.join(root_path, seq_name,
                                       seq_name + '_outline_' + str(init_method_name) + '_CSS_proto.pkl')

        with open(proto_info_path, 'rb') as f:
            proto_set = pickle.load(f)

        proto_points_set = proto_set['proto_points_set']

        valid_proto_id = []
        valid_names = []

        box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                    torch.from_numpy(outline_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
                ).long().squeeze(dim=0).cpu().numpy()

        retain_mask = np.ones(shape = (points.shape[0],))

        for i in range(len(outline_boxes)):
            this_box = outline_boxes[i]
            this_name = outline_cls[i]
            this_score = outline_score[i]
            this_proto_id = proto_id[i]

            if this_name == 'Dis_Small':
                retain_mask*=(box_idxs_of_pts!=i)
            elif this_name in ['Vehicle', 'Pedestrian', 'Cyclist']:

                if this_score > dataset_cfg.RefinerConfig.DiscardThresh[this_name]:
                    new_outline_boxes.append(this_box)
                    new_outline_cls.append(this_name)
                    new_proto_id.append(this_proto_id)
                    new_outline_score.append(this_score)
                    if this_proto_id not in valid_proto_id and this_proto_id >= 0:
                        valid_proto_id.append(this_proto_id)
                        valid_names.append(this_name)
                else:
                    retain_mask*=(box_idxs_of_pts!=i)

        points = points[retain_mask.astype(bool)]

        all_valid_proto_points = []
        all_valid_proto_boxes = []

        for i in range(len(valid_proto_id)):
            this_proto = proto_points_set[valid_names[i]][valid_proto_id[i]]
            this_proto_points = this_proto['points']
            this_proto_box = this_proto['box']
            this_proto_score = 1
            this_proto_id = valid_proto_id[i]

            

            random_pose = self.compute_proto_pos(new_outline_boxes)

            this_proto_points[:, 0:2] -= this_proto_box[0:2]
            this_proto_points[:, 0:2] += random_pose[...]
            this_proto_box[0:2] = random_pose[...]

            new_points = np.zeros(shape=(this_proto_points.shape[0], points.shape[1]))
            new_points[:, 0:3] = this_proto_points[:, 0:3]

            all_valid_proto_points.append(new_points)
            all_valid_proto_boxes.append(this_proto_box)

            new_outline_boxes.append(this_proto_box)
            new_outline_cls.append(valid_names[i])
            new_proto_id.append(this_proto_id)
            new_outline_score.append(this_proto_score)

        
        all_valid_proto_boxes = np.array(all_valid_proto_boxes)

        if len(all_valid_proto_boxes)>0:
            all_valid_proto_points = np.concatenate(all_valid_proto_points, 0)
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                    torch.from_numpy(all_valid_proto_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
                ).long().squeeze(dim=0).cpu().numpy()

            points = points[box_idxs_of_pts==(-1)]
            points = np.concatenate([points, all_valid_proto_points])


        if np.random.randint(2):
            randoms = np.random.permutation(len(points))
            points = points[randoms[0:int(len(points)*0.1)]]
            #points = la_sampling(points, vert_res=0.003)

        return points, np.array(new_outline_boxes), \
               np.array(new_outline_cls), \
               np.array(new_outline_score), \
               np.array(new_proto_id),


    def sample_prototype_cpu(self, seq_name,
                       root_path,
                       points,
                       outline_boxes,
                       outline_cls,
                       outline_score,
                       proto_id,
                       dataset_cfg):

        new_outline_boxes = []
        new_outline_cls = []
        new_outline_score = []
        new_proto_id = []

        init_method_name = dataset_cfg.InitLabelGenerator
        proto_info_path = os.path.join(root_path, seq_name,
                                       seq_name + '_outline_' + str(init_method_name) + '_CSS_proto.pkl')

        with open(proto_info_path, 'rb') as f:
            proto_set = pickle.load(f)

        proto_points_set = proto_set['proto_points_set']

        valid_proto_instance_points = []

        # box_idxs_of_pts = cut_all_boxes(points[:, 0:3], outline_boxes[:, 0:7])
        box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_cpu(
            points[:, 0:3],
            outline_boxes[:, 0:7]
        )

        retain_mask_no_object = np.ones(shape=(points.shape[0],))

        retain_mask_good_object = np.ones(shape=(points.shape[0],))

        for i in range(len(outline_boxes)):
            this_box = outline_boxes[i]
            this_name = outline_cls[i]
            this_score = outline_score[i]
            this_proto_id = proto_id[i]

            retain_mask_no_object *= (box_idxs_of_pts[i] == 0)
            if this_name == 'Dis_Small':
                retain_mask_good_object *= (box_idxs_of_pts[i] == 0)
            elif this_name in ['Vehicle', 'Pedestrian', 'Cyclist']:

                if this_score > dataset_cfg.RefinerConfig.DiscardThresh[this_name] and np.linalg.norm(
                        this_box[0:2]) < 75 and this_proto_id >= 0:
                    new_outline_boxes.append(this_box)
                    new_outline_cls.append(this_name)
                    new_proto_id.append(this_proto_id)
                    new_outline_score.append(this_score)

                    this_proto = proto_points_set[this_name][this_proto_id]

                    this_proto_points = this_proto['points']
                    this_proto_box = this_proto['box']

                    box_idxs_of_pts_proto = roiaware_pool3d_utils.points_in_boxes_cpu(
                        this_proto_points[:, 0:3],
                        np.array([this_proto_box])
                    )

                    this_proto_points = this_proto_points[box_idxs_of_pts_proto[0]!=0]

                    cloud = np.zeros(shape=(this_proto_points.shape[0], 4))
                    cloud[:, 0:3] = this_proto_points[:, 0:3]
                    cloud[:, 3] = 1

                    x, y, z, yaw = this_proto_box[0], this_proto_box[1], this_proto_box[2], this_proto_box[6]

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

                    x, y, z, yaw = this_box[0], this_box[1], this_box[2], this_box[6]
                    trans_mat = np.eye(4, dtype=np.float32)
                    trans_mat[0, 0] = np.cos(yaw)
                    trans_mat[0, 1] = -np.sin(yaw)
                    trans_mat[0, 3] = x
                    trans_mat[1, 0] = np.sin(yaw)
                    trans_mat[1, 1] = np.cos(yaw)
                    trans_mat[1, 3] = y
                    trans_mat[2, 3] = z

                    cloud = np.matmul(cloud, trans_mat.T)
                    cloud_new = np.zeros(shape=(cloud.shape[0], points.shape[1]))
                    cloud_new[:, 0:3] = cloud[:, 0:3]

                    valid_proto_instance_points.append(cloud_new)
                else:
                    retain_mask_good_object *= (box_idxs_of_pts[i] == 0)
            else:
                retain_mask_good_object *= (box_idxs_of_pts[i] == 0)

        points_no_obj = points[retain_mask_no_object.astype(bool)]
        points_good_obj = points[retain_mask_good_object.astype(bool)]

        points_proto = np.concatenate(valid_proto_instance_points + [points_no_obj], 0)

        if np.random.randint(2):
            # points_good_obj = la_sampling(points_proto, vert_res = 0.003)
            #
            # new_proto_pts = []
            #
            # for pts_proto in valid_proto_instance_points:
            #     randoms = np.random.permutation(len(pts_proto))
            #     rate = np.random.random() * 0.01
            #     if rate < 0.001:
            #         rate = 0.001
            #     new_points = pts_proto[randoms[0:int(len(pts_proto) * rate)]]
            #     if len(new_points) == 0:
            #         new_proto_pts.append(pts_proto[randoms[0:5]])
            #     else:
            #         new_proto_pts.append(new_points)
            # randoms = np.random.permutation(len(points_no_obj))
            # new_points_no_obj = points_no_obj[randoms[0:int(len(points_no_obj) * 0.2)]]
            # points_proto = np.concatenate(new_proto_pts + [new_points_no_obj], 0)

            randoms = np.random.permutation(len(points_good_obj))
            points_good_obj = points_good_obj[randoms[0:int(len(points_good_obj) * 0.2)]]

        return points_good_obj, points_proto, np.array(new_outline_boxes), \
               np.array(new_outline_cls), \
               np.array(new_outline_score), \
               np.array(new_proto_id),

    def get_frame(self, index):

        num_frames = self.num_data_frames
        info = copy.deepcopy(self.all_infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']

        all_points = []
        cur_pose = np.linalg.inv(info['pose'])

        for i in range(index-num_frames+1, index+1):

            if i>=0:
                this_info = self.all_infos[i]
                this_sample_idx = this_info['point_cloud']['sample_idx']
                this_seq_idx = this_info['point_cloud']['lidar_sequence']
                if this_seq_idx==sequence_name:
                    this_points = self.get_lidar(this_seq_idx, this_sample_idx)
                    this_points[:, 3] = 0

                    this_points[:, 0:3] = self.points_rigid_transform(this_points[:, 0:3], this_info['pose'])
                    this_points[:, 0:3] = self.points_rigid_transform(this_points[:, 0:3], cur_pose)
                    this_points[:, -1] = 0 #i - (index-num_frames)

                    all_points.append(this_points)

        points = np.concatenate(all_points)

        current_frame_id = self.all_infos[index]['point_cloud']['sample_idx']
        current_seq_id = self.all_infos[index]['point_cloud']['lidar_sequence']

        input_dict = {

            'frame_id': current_frame_id,
            'seq_id': current_seq_id
        }
        proto_points = copy.deepcopy(points)


        current_label_method = self.dataset_cfg.current_label_method
        labeling_methods = self.dataset_cfg.labeling_method

        if self.dataset_cfg.current_label_method == 'unlabeled' and self.training:

            gt_boxes = info['outline_box']
            names = info['outline_cls']
            if self.dataset_cfg.get('LabelRefiner', None) in ['C_PROTO','C_PROTO_SI'] :
                score = info['outline_score']
                proto_id = info['outline_proto_id']


                points, proto_points, gt_boxes, names, score, proto_id = self.sample_prototype_cpu(sequence_name,
                                                                        self.data_path,
                                                                        points,
                                                                        gt_boxes,
                                                                        names,
                                                                        score,
                                                                        proto_id,
                                                                        self.dataset_cfg)

                if len(gt_boxes)==0:

                    input_dict.update({
                        'gt_names': np.empty(shape=(0,)),
                        'gt_boxes': np.empty(shape=(0,7)),
                        'num_points_in_gt': np.empty(shape=(0,)),
                        'css_score': np.empty(shape=(0,)),
                        'proto_group_id': np.empty(shape=(0,))
                    })
                else:
                    input_dict.update({
                        'gt_names': names,
                        'gt_boxes': gt_boxes,
                        'num_points_in_gt': np.ones((gt_boxes.shape[0]))*100,
                        'css_score': score,
                        'proto_group_id': proto_id
                    })
            else:
                if len(gt_boxes)<=1:

                    input_dict.update({
                        'gt_names': np.empty(shape=(0,)),
                        'gt_boxes': np.empty(shape=(0,7)),
                        'num_points_in_gt': np.empty(shape=(0,)),
                    })
                else:
                    input_dict.update({
                        'gt_names': names,
                        'gt_boxes': gt_boxes,
                        'num_points_in_gt': np.ones((gt_boxes.shape[0]))*100,
                    })



        elif current_label_method in labeling_methods and self.training:
            valid_names = []
            valid_boxes = []
            valid_num = []
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='unknown')
            for cls in self.class_names:
                cls_mask = annos['name']==cls
                name = annos['name'][cls_mask]
                box = annos['gt_boxes_lidar'][cls_mask]
                num_points_in_gt = annos['num_points_in_gt'][cls_mask]

                for i in range(len(box)):
                    if index%labeling_methods[current_label_method]['frame_interval']==0\
                        and i<labeling_methods[current_label_method]['label_num_per_frame']:
                        valid_names.append(name[i])
                        valid_boxes.append(box[i])
                        valid_num.append(num_points_in_gt[i])

            if len(valid_names)==0:
                input_dict.update({
                    'gt_names': np.empty(shape=(0,)),
                    'gt_boxes': np.empty(shape=(0, 7)),
                    'num_points_in_gt': np.empty(shape=(0,))
                })
            else:
                input_dict.update({
                    'gt_names': np.array(valid_names),
                    'gt_boxes': np.array(valid_boxes),
                    'num_points_in_gt': np.array(valid_num)
                })

        elif 'annos' in info and (not self.training):
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='unknown')

            if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']

            num_points_in_gt = annos.get('num_points_in_gt', None)

            if num_points_in_gt is not None:
                mask = num_points_in_gt[:]>0
                annos['name'] = annos['name'][mask]
                gt_boxes_lidar = gt_boxes_lidar[mask]
                num_points_in_gt = num_points_in_gt[mask]

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar,
                'num_points_in_gt': num_points_in_gt
            })

        input_dict['points'] = points
        input_dict['points1'] = proto_points

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['metadata'] = info.get('metadata', info['frame_id'])
        data_dict.pop('num_points_in_gt', None)

        input_dict['points'][:,3:]=0
        input_dict['points1'][:,3:]=0

        return data_dict


    def __getitem__(self, index):

        index = self.index_list[index]

        data_dict=self.get_frame(index)
        return data_dict

    #@staticmethod
    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            mask_car = pred_labels[:]==1

            if mask_car.sum()!=0:

                pred_boxes[mask_car, 2] = pred_boxes[mask_car, 2] + self.dataset_cfg.get('LABEL_OFFSET', 0)

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)

            if self.test_augmentor is not None:
                single_pred_dict = self.test_augmentor.backward(single_pred_dict)

            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['seq_id'] = batch_dict['seq_id'][index]
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                'Vehicle': 'Car',
                'Pedestrian': 'Pedestrian',
                'Cyclist': 'Cyclist',
                'Sign': 'Sign',
                'Car': 'Car'
            }
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from .waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)

        eval_gt_annos = []

        for i in range(len(self.infos)):
            info = copy.deepcopy(self.infos[i])
            gt_annos = info['annos']
            frame_id = info['point_cloud']['sample_idx']
            seq_id = info['point_cloud']['lidar_sequence']
            gt_annos['frame_id'] = frame_id
            gt_annos['seq_id'] = seq_id
            eval_gt_annos.append(gt_annos)

        #eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
        elif kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def map_to_real_label(self,
                          outline_box,
                          outline_ids,
                          outline_cls,
                          is_str = False):
        names = []
        gt_boxes = []
        obj_ids = []
        difficulty = []

        for i in range(len(outline_cls)):
            if is_str:
                names.append(outline_cls[i])
                gt_boxes.append(outline_box[i])
                obj_ids.append(outline_ids[i])
                difficulty.append(1)
            elif str(outline_cls[i]) in self.dataset_cfg.map_to_real_label:
                name = self.dataset_cfg.map_to_real_label[str(outline_cls[i])]
                names.append(name)
                gt_boxes.append(outline_box[i])
                obj_ids.append(outline_ids[i])
                difficulty.append(1)

        if len(gt_boxes)==0:
            return np.empty(shape=(0,)), np.empty(shape=(0,7)), np.empty(shape=(0,)), np.empty(shape=(0,))
        else:
            return np.array(names), np.array(gt_boxes), np.array(obj_ids), np.array(difficulty)

    def create_track_groundtruth_database(self, info_path, save_path, used_classes=None, split='train'):

        labeling_method = self.dataset_cfg.labeling_method

        gt_path_name = Path('pcdet_gt_track_database_%s_cp' % split)

        database_save_path = save_path / gt_path_name

        db_info_save_path = save_path / ('pcdet_waymo_track_dbinfos_%s_cp.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)

        all_db_infos = {}

        for cls_name in used_classes:
            all_db_infos[cls_name] = []

        infos = self.all_infos

        for cls_name in used_classes:

            for k in range(0, len(infos)):

                if (cls_name == 'Vehicle') and k%10!=0:
                    continue
                if (cls_name == 'Pedestrian') and k%5!=0:
                    continue

                print(cls_name+'_tracks_gt_database sample: %d/%d' % (k + 1, len(infos)))

                info = copy.deepcopy(infos[k])

                pc_info = info['point_cloud']
                sequence_name = pc_info['lidar_sequence']
                sample_idx = pc_info['sample_idx']
                points = self.get_lidar(sequence_name, sample_idx)


                outline_box = info['outline_box']
                outline_ids = info['outline_ids']
                outline_cls = info['outline_cls']

                names, gt_boxes, obj_ids, difficulty = self.map_to_real_label(outline_box,
                                                                              outline_ids,
                                                                              outline_cls,
                                                                              is_str=True)

                mask = (names == cls_name)

                names = names[mask]
                gt_boxes = gt_boxes[mask]
                difficulty = difficulty[mask]
                obj_ids = obj_ids[mask]

                num_obj = gt_boxes.shape[0]
                if num_obj == 0:
                    continue

                box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                    torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
                ).long().squeeze(dim=0).cpu().numpy()

                for i in range(num_obj):

                    labeling_method_dict = ['unlabeled']

                    ob_id = obj_ids[i]
                    filename = '%s_%s.bin' % (names[i], ob_id)
                    filepath = database_save_path / sequence_name / str(sample_idx) / filename

                    gt_points = points[box_idxs_of_pts == i]
                    gt_points[:, :3] -= gt_boxes[i, :3]

                    if gt_points.shape[0] <= 5:
                        continue

                    if (used_classes is None) or names[i] in used_classes:
                        with open(filepath, 'w') as f:
                            gt_points.tofile(f)


                    if (used_classes is None) or names[i] in used_classes:
                        db_path = str(
                            gt_path_name / sequence_name / str(sample_idx) / filename)  # gt_database/xxxxx.bin

                        db_info = {'name': cls_name, 'path': db_path, 'sequence_name': sequence_name,
                                   "seq_idx": sequence_name, 'image_idx': sample_idx, 'sample_idx': sample_idx,
                                   'gt_idx': i, 'ob_idx': ob_id,
                                   'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                                   'pose': info['pose'],
                                   'difficulty': difficulty[i],
                                   'labeling_method_dict': labeling_method_dict}

                        all_db_infos[cls_name].append(db_info)


            for k, v in all_db_infos.items():
                print('Database %s: %d' % (k, len(v)))

            with open(db_info_save_path, 'wb') as f:
                pickle.dump(all_db_infos, f)

    def create_ppscore(self, root_path):

        sample_sequence_file_list = [ os.path.splitext(os.path.basename(sequence_file))[0]
                                      for sequence_file in self.sample_sequence_list]

        process_single_sequence = partial(
            save_pp_score,
            root_path=str(root_path),
            max_win=self.dataset_cfg.PPScoreConfig.max_win_size,
            win_inte=self.dataset_cfg.PPScoreConfig.win_interval,
            max_neighbor_dist=self.dataset_cfg.PPScoreConfig.max_neighbor_dist
        )

        with multiprocessing.Pool(16) as p:
            sequence_infos = list(tqdm.tqdm(p.imap(process_single_sequence, sample_sequence_file_list), total=len(sample_sequence_file_list)))

        return sequence_infos

    def create_outline_box(self, root_path):
        sample_sequence_file_list = [ os.path.splitext(os.path.basename(sequence_file))[0]
                                      for sequence_file in self.sample_sequence_list]

        process_single_sequence = partial(
            compute_outline_box,
            root_path=str(root_path),
            dataset_cfg=self.dataset_cfg,
        )

        with multiprocessing.Pool(16) as p:
            sequence_infos = list(tqdm.tqdm(p.imap(process_single_sequence, sample_sequence_file_list), total=len(sample_sequence_file_list)))

        sequences_infos = [item for infos in sequence_infos for item in infos]

        return sequences_infos


def create_waymo_infos(dataset_cfg, class_names, data_path, save_path,
                       raw_data_tag='raw_data', processed_data_tag='waymo_processed_data',
                       workers=multiprocessing.cpu_count()):
    # multiprocessing.cpu_count()
    # https://blog.csdn.net/qq_30159015/article/details/82658896

    dataset = WaymoUnsupervisedDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split, test_split = 'train', 'val', 'test'

    train_filename = save_path / ('waymo_infos_%s.pkl' % train_split)
    val_filename = save_path / ('waymo_infos_%s.pkl' % val_split)
    test_filename = save_path / ('waymo_infos_%s.pkl' % test_split)

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)

    waymo_infos_train = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1
    )

    
    if dataset.dataset_cfg.get('PPScoreConfig', None) is not None:
        print('---------------Start to generate ppscore---------------')
        dataset.create_ppscore(root_path=save_path / processed_data_tag)


    if dataset.dataset_cfg.get('MERGE_CONFIG', None) is not None:
        if os.path.exists(os.path.join(dataset.dataset_cfg.MERGE_CONFIG.DET_PATH, 'result_merged.pkl')):
            print('detections are merged before !!!')
        else:
            merger = DetectionsMerger(dataset.dataset_cfg)
            merger.merge()

    
    print('---------------Start to create outline box---------------')
    waymo_infos_train = dataset.create_outline_box(root_path=save_path / processed_data_tag)

    outline_info_name = ''

    if 'InitLabelGenerator' in dataset_cfg:
        outline_info_name = dataset_cfg.InitLabelGenerator

    if 'LabelRefiner' in dataset_cfg:
        outline_info_name = dataset_cfg.LabelRefiner
    train_filename = save_path / (outline_info_name+'_waymo_infos_%s.pkl' % train_split)

    with open(train_filename, 'wb') as f:
        pickle.dump(waymo_infos_train, f)
    print('----------------Waymo info train file is saved to %s----------------' % train_filename)
    
    dataset.set_split(val_split)
    waymo_infos_val = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1
    )

    with open(val_filename, 'wb') as f:
        pickle.dump(waymo_infos_val, f)
    print('----------------Waymo info val file is saved to %s----------------' % val_filename)


    dataset.set_split(test_split)
    waymo_infos_test = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=False,
        sampled_interval=1
    )

    with open(test_filename, 'wb') as f:
        pickle.dump(waymo_infos_test, f)
    print('----------------Waymo info val file is saved to %s----------------' % test_filename)
    

    
    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
   
    dataset.create_track_groundtruth_database(
        info_path=train_filename, save_path=save_path, split='train',
        used_classes=['Vehicle', 'Pedestrian', 'Cyclist']
    )
    print('---------------Data preparation Done---------------')
    


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='waymo_unsupervised_cproto.yaml', help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_waymo_infos', help='')
    args = parser.parse_args()

    if args.func == 'create_waymo_infos':
        import yaml
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        ROOT_DIR = Path(ROOT_DIR)
        ROOT_DIR = ROOT_DIR.resolve()
        create_waymo_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR,
            save_path=ROOT_DIR,
            raw_data_tag='raw_data',
            processed_data_tag=dataset_cfg.PROCESSED_DATA_TAG
        )

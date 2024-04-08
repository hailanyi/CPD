import os
import pickle as pkl
from cpd.unsupervised_core.outline_utils import OutlineFitter, TrackSmooth, voxel_sampling, points_rigid_transform
import numpy as np

class MFCF():
    def __init__(self, seq_name, root_path, config):
        self.seq_name = seq_name
        self.root_path = root_path
        self.dataset_cfg = config

        self.outline_estimator = OutlineFitter(sensor_height=config.GeneratorConfig.sensor_height,
                 ground_min_threshold=config.GeneratorConfig.ground_min_threshold,
                 ground_min_distance=config.GeneratorConfig.ground_min_distance,
                 cluster_dis =config.GeneratorConfig.cluster_dis,
                 cluster_min_points=config.GeneratorConfig.cluster_min_points,
                 discard_max_height=config.GeneratorConfig.discard_max_height,
                 min_box_volume=config.GeneratorConfig.min_box_volume,
                 min_box_height=config.GeneratorConfig.min_box_height,
                 max_box_volume=config.GeneratorConfig.max_box_volume,
                 max_box_len=config.GeneratorConfig.max_box_len)

    def generate_outline_box(self,):
        seq_name, root_path, dataset_cfg = self.seq_name, self.root_path, self.dataset_cfg

        input_pkl_path = os.path.join(root_path, seq_name, seq_name + '.pkl')

        method_name = dataset_cfg.InitLabelGenerator

        output_pkl_path = os.path.join(root_path, seq_name, seq_name + '_outline_'+str(method_name)+'.pkl')

        if os.path.exists(output_pkl_path):
            with open(output_pkl_path, 'rb') as f:
                infos = pkl.load(f)
            return infos

        with open(input_pkl_path, 'rb') as f:
            infos = pkl.load(f)

        all_labels = []
        all_pose = []
        win_size = self.dataset_cfg.GeneratorConfig.frame_num
        inte = self.dataset_cfg.GeneratorConfig.frame_interval
        thresh = self.dataset_cfg.GeneratorConfig.ppscore_thresh

        for i in range(0, len(infos)):

            pose_i = np.linalg.inv(infos[i]['pose'])
            all_points = []
            all_H = []
            cur_points = None

            for j in range(i - win_size, i + win_size, inte):
                info_path = str(j).zfill(4) + '.npy'
                lidar_path = os.path.join(root_path, seq_name, info_path)
                if not os.path.exists(lidar_path):
                    continue
                pose_j = infos[j]['pose']
                lidar_points = np.load(lidar_path)[:, 0:3]
                if j == i:
                    cur_points = lidar_points

                lidar_points = points_rigid_transform(lidar_points, pose_j)
                H_path = os.path.join(root_path, seq_name, 'ppscore', info_path)
                H = np.load(H_path)
                all_points.append(lidar_points)
                all_H.append(H)
            all_points = np.concatenate(all_points)
            all_points = points_rigid_transform(all_points, pose_i)
            all_H = np.concatenate(all_H)
            all_points = all_points[all_H > thresh]
            new_box_points = np.concatenate([all_points, cur_points])
            new_box_points = voxel_sampling(new_box_points)

            non_ground_points = self.outline_estimator.remove_ground(new_box_points)
            clusters, labels = self.outline_estimator.clustering(non_ground_points)
            boxes = self.outline_estimator.box_fit_DGD(clusters)

            all_labels.append(boxes)
            all_pose.append(infos[i]['pose'])

        tracker = TrackSmooth(self.dataset_cfg.GeneratorConfig)
        tracker.tracking(all_labels, all_pose)

        for i in range(0, len(infos)):

            objs, ids, cls, dif = tracker.get_current_frame_objects_and_cls(i)

            infos[i]['outline_box'] = objs
            infos[i]['outline_ids'] = ids
            infos[i]['outline_cls'] = cls
            infos[i]['outline_dif'] = dif

        with open(output_pkl_path, 'wb') as f:
            pkl.dump(infos, f)

        return infos

    def __call__(self, ):
        suc = self.generate_outline_box()

        return suc
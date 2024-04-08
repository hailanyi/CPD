import os
import pickle as pkl
from pcdet.unsupervised_core.outline_utils import OutlineFitter, drop_cls
import numpy as np

class PPSCORE():
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

        for i, info in enumerate(infos):

            info_path = str(i).zfill(4) + '.npy'
            lidar_path = os.path.join(root_path, seq_name, info_path)
            lidar_points = np.load(lidar_path)[:, 0:3]

            H_path = os.path.join(root_path, seq_name, 'ppscore', info_path)
            H = np.load(H_path)

            lidar_points = lidar_points[H<self.dataset_cfg.GeneratorConfig.ppscore_thresh]

            non_ground_points = self.outline_estimator.remove_ground(lidar_points)
            clusters, labels = self.outline_estimator.clustering(non_ground_points)
            boxes = self.outline_estimator.box_fit(clusters)

            boxes, cls, dif = self.outline_estimator.get_box_cls(boxes, dataset_cfg.GeneratorConfig)

            boxes, cls, _, dif, _, _ = drop_cls(boxes, cls, dif=dif)

            infos[i]['outline_box'] = boxes
            infos[i]['outline_cls'] = cls
            infos[i]['outline_dif'] = dif




        with open(output_pkl_path, 'wb') as f:
            pkl.dump(infos, f)

        return infos

    def __call__(self, ):
        suc = self.generate_outline_box()
        return suc
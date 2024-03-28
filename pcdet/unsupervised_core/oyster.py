import os
import pickle as pkl
from pcdet.unsupervised_core.outline_utils import OutlineFitter, TrackSmooth, drop_cls, corner_align
import numpy as np
import copy

class OYSTER():
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

        input_pkl_path2 = os.path.join(root_path, seq_name, seq_name +'_outline_MFCF.pkl')

        method_name = dataset_cfg.InitLabelGenerator

        output_pkl_path = os.path.join(root_path, seq_name, seq_name + '_outline_'+str(method_name)+'.pkl')
        '''
        if os.path.exists(output_pkl_path):
            with open(output_pkl_path, 'rb') as f:
                infos = pkl.load(f)
            return infos
        '''
        if os.path.exists(input_pkl_path2):
            with open(input_pkl_path2, 'rb') as f:
                infos = pkl.load(f)
        else:
            with open(input_pkl_path, 'rb') as f:
                infos = pkl.load(f)


        all_labels = []
        all_pose = []

        for i in range(0, len(infos)):

            if 'outline_box' in infos[i]:
                boxes = infos[i]['outline_box']
            else:
                info_path = str(i).zfill(4) + '.npy'
                lidar_path = os.path.join(root_path, seq_name, info_path)
                lidar_points = np.load(lidar_path)[:, 0:3]
                non_ground_points = self.outline_estimator.remove_ground(lidar_points)
                clusters, labels = self.outline_estimator.clustering(non_ground_points)
                boxes = self.outline_estimator.box_fit(clusters)

            all_labels.append(boxes)
            all_pose.append(infos[i]['pose'])

        tracker = TrackSmooth(self.dataset_cfg.GeneratorConfig)
        tracker.tracking(all_labels, all_pose)


        # to track id first dict
        trajectory = {}

        for i in range(0, len(infos)):

            objs, ids, cls, dif = tracker.get_current_frame_objects_and_cls(i)

            objs, cls, ids, dif, _, _ = drop_cls(objs, cls, dif=dif, ids=ids)

            if len(ids)<=1:
                continue

            for j, id in enumerate(ids):
                if id in trajectory:
                    trajectory[id][i] = [objs[j],cls[j],dif[j]]
                else:
                    trajectory[id] = {i:[objs[j],cls[j],dif[j]]}


        # corner align in track
        for id in trajectory:

            this_track = trajectory[id]
            if len(this_track)<6:
                continue

            all_objects = []
            all_t_id = []

            for t_id in this_track.keys():
                all_objects.append(this_track[t_id][0])
                all_t_id.append(t_id)
            all_objects = np.array(all_objects)
            objects_dis = np.linalg.norm(np.array(all_objects)[:, 0:3], axis=-1)

            arg_min = np.argsort(objects_dis)
            top_len = int(len(arg_min)*(1-0.95))
            if top_len<=3:
                top_len=3
            new_objects_sort = copy.deepcopy(all_objects)[arg_min]
            top_objects = new_objects_sort[0:top_len]
            mean_whl = np.mean(top_objects, axis=0)

            for this_i, box in enumerate(all_objects):
                new_box = corner_align(box, mean_whl[3]-box[3], mean_whl[4]-box[4])
                this_track[all_t_id[this_i]][0] = new_box

        # to frame first dict
        frame_first_dict = {}

        for id in trajectory:

            this_track = trajectory[id]
            if len(this_track)<6:      # filter low confidence track
                continue

            for t_id in this_track.keys():
                if t_id in frame_first_dict:
                    frame_first_dict[t_id]['outline_box'].append(this_track[t_id][0])
                    frame_first_dict[t_id]['outline_ids'].append(id)
                    frame_first_dict[t_id]['outline_cls'].append(this_track[t_id][1])
                    frame_first_dict[t_id]['outline_dif'].append(this_track[t_id][2])
                else:
                    frame_first_dict[t_id] = {'outline_box':[this_track[t_id][0]],
                                              'outline_ids':[id],
                                              'outline_cls':[this_track[t_id][1]],
                                              'outline_dif':[this_track[t_id][2]],}

        for i in range(0, len(infos)):
            if i in frame_first_dict:
                infos[i]['outline_box'] = np.array(frame_first_dict[i]['outline_box'])
                infos[i]['outline_ids'] = np.array(frame_first_dict[i]['outline_ids'])
                infos[i]['outline_cls'] = np.array(frame_first_dict[i]['outline_cls'])
                infos[i]['outline_dif'] = np.array(frame_first_dict[i]['outline_dif'])
            else:
                infos[i]['outline_box'] = np.empty(shape=(0, 7))
                infos[i]['outline_ids'] = np.empty(shape=(0,))
                infos[i]['outline_cls'] = np.empty(shape=(0,))
                infos[i]['outline_dif'] = np.empty(shape=(0,))

        with open(output_pkl_path, 'wb') as f:
            pkl.dump(infos, f)


        return infos

    def __call__(self, ):
        suc = self.generate_outline_box()
        return suc
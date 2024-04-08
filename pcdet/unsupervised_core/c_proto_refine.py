import numpy as np
import copy
import os
import pickle as pkl

from cpd.unsupervised_core.outline_utils import OutlineFitter, points_rigid_transform, correct_heading,\
    hierarchical_occupancy_score, smooth_points, angle_from_vector, get_registration_angle, box_rigid_transform,\
    correct_orientation, density_guided_drift,\
    KL_entropy_score

from cpd.unsupervised_core.ob_op import box_cut

class CSS():
    def __init__(self, config):
        self.max_dis = config.MaxDis
        self.mlo_parts = config.MLOParts
        self.predifined_size = config.PredifinedSize
        self.weights = np.array(config.CSS_weight)

    def compute_css(self, points, box, name):

        predefined_size = self.predifined_size[name] # (3)

        dis_dis = np.linalg.norm(box[0:3])
        if dis_dis > self.max_dis:
            dis_dis = self.max_dis
        dis_score = 1 - dis_dis / self.max_dis

        mlo_score = hierarchical_occupancy_score(points, box, self.mlo_parts)

        new_box = copy.deepcopy(box)
        this_size_norm = new_box[3:6] / new_box[3:6].sum()
        this_temp_norm = np.array(predefined_size)
        this_temp_norm = this_temp_norm / this_temp_norm.sum()
        size_score = KL_entropy_score(this_size_norm, this_temp_norm)

        weights = np.array(self.weights)/np.sum(self.weights)

        final_score = dis_score*weights[0]+mlo_score*weights[1]+size_score*weights[2]

        return final_score

    def __call__(self, points, box, name):
        return self.compute_css(points, box, name)

class C_PROTO():
    def __init__(self, seq_name, root_path, config):
        self.seq_name = seq_name
        self.root_path = root_path
        self.dataset_cfg = config

        self.outline_estimator = OutlineFitter(sensor_height=config.GeneratorConfig.sensor_height,
                 ground_min_threshold=config.RefinerConfig.GroundMin,
                 ground_min_distance=config.GeneratorConfig.ground_min_distance,
                 cluster_dis =config.GeneratorConfig.cluster_dis,
                 cluster_min_points=config.GeneratorConfig.cluster_min_points,
                 discard_max_height=config.GeneratorConfig.discard_max_height,
                 min_box_volume=config.GeneratorConfig.min_box_volume,
                 min_box_height=config.GeneratorConfig.min_box_height,
                 max_box_volume=config.GeneratorConfig.max_box_volume,
                 max_box_len=config.GeneratorConfig.max_box_len)

        self.css_estimator = CSS(config.RefinerConfig.CSSConfig)

    def compute_css_score_and_raw_proto(self):
        seq_name, root_path, dataset_cfg = self.seq_name, self.root_path, self.dataset_cfg

        init_method_name = dataset_cfg.InitLabelGenerator
        input_pkl_path = os.path.join(root_path, seq_name, seq_name + '_outline_'+str(init_method_name)+'.pkl')

        output_pkl_path = os.path.join(root_path, seq_name, seq_name + '_outline_'+str(init_method_name)+'_CSS.pkl')

        output_raw_proto_path = os.path.join(root_path, seq_name, seq_name + '_outline_' + str(init_method_name) + '_CSS_raw_proto.pkl')

        raw_proto_set = {'Vehicle': {},
                         'Pedestrian': {},
                         'Cyclist': {} }

        basic_proto_score_thresh = self.dataset_cfg.RefinerConfig.BasicProtoScoreThresh
        seq_id = int(self.seq_name[8:16])

        if os.path.exists(output_pkl_path):
            with open(output_pkl_path, 'rb') as f:
                infos = pkl.load(f)
            return infos

        with open(input_pkl_path, 'rb') as f:
            outline_infos = pkl.load(f)


        for i in range(len(outline_infos)):

            outline_box = outline_infos[i]['outline_box']
            outline_cls = outline_infos[i]['outline_cls']
            outline_ids = outline_infos[i]['outline_ids']
            outline_score = np.zeros(shape=outline_cls.shape)
            pose = outline_infos[i]['pose']

            info_path = str(i).zfill(4) + '.npy'
            lidar_path = os.path.join(root_path, seq_name, info_path)
            lidar_points = np.load(lidar_path)[:, 0:3]

            for box_id in range(len(outline_box)):
                this_box = outline_box[box_id]
                this_box_name = outline_cls[box_id]
                this_track_id = outline_ids[box_id]

                if this_box_name not in raw_proto_set:
                    continue

                if this_box_name=='Pedestrian':
                    pre_size = np.array(self.css_estimator.predifined_size['Pedestrian'])
                    this_box[3:5] = pre_size[0:2]
                    outline_infos[i]['outline_box'][box_id] = this_box
                if this_box_name=='Cyclist':
                    pre_size = np.array(self.css_estimator.predifined_size['Cyclist'])
                    this_box[4] = pre_size[1]
                    outline_infos[i]['outline_box'][box_id] = this_box

                dis = np.sqrt(np.sum((lidar_points[:, 0:2] - this_box[0:2]) ** 2, -1))
                mask_low = dis < (max(this_box[3], this_box[4]))

                this_low_points = lidar_points[mask_low]

                if len(this_low_points) > 0:
                    this_low_points = smooth_points(this_low_points)

                if len(this_low_points) > 0:
                    z_min = min(this_low_points[:, 2])
                else:
                    z_min = this_box[2] - this_box[5] / 2

                z_max = this_box[2] + this_box[5] / 2

                h = z_max - z_min
                if h < 1.3:
                    h = 1.3
                z = h / 2 + z_min

                new_box = np.array([this_box[0], this_box[1], z, this_box[3], this_box[4], h, this_box[6]])

                if len(this_low_points) > 0:

                    valid_low_mask = this_low_points[:, 2] > z_min + 0.2
                    valid_high_mask = this_low_points[:, 2] < z_max
                    mask = valid_low_mask * valid_high_mask
                    this_low_points = this_low_points[mask]

                    non_ground_points = self.outline_estimator.remove_ground(this_low_points)

                    if len(non_ground_points) > 10:
                        clusters, labels = self.outline_estimator.clustering(non_ground_points)
                        if len(clusters) > 0:
                            max_cluter = clusters[0]
                            for clu in clusters:
                                if len(clu) > len(max_cluter):
                                    max_cluter = clu

                            css_score = self.css_estimator(max_cluter, new_box, this_box_name)

                            outline_score[box_id] = css_score
                            outline_infos[i]['outline_box'][box_id] = new_box

                            if css_score>basic_proto_score_thresh[this_box_name]:
                                proto_id = int(str(seq_id)+str(this_track_id))

                                global_position = points_rigid_transform([new_box[0:3]], pose)[0:,0:3]

                                if proto_id in raw_proto_set[this_box_name]:

                                    pose_i = np.linalg.inv(raw_proto_set[this_box_name][proto_id]['pose'][0])

                                    max_cluter_global = points_rigid_transform(max_cluter, pose)
                                    points_in_first = points_rigid_transform(max_cluter_global, pose_i)

                                    raw_proto_set[this_box_name][proto_id]['points'].append(points_in_first)
                                    raw_proto_set[this_box_name][proto_id]['outline_box'].append(new_box)
                                    raw_proto_set[this_box_name][proto_id]['pose'].append(pose)
                                    raw_proto_set[this_box_name][proto_id]['score'].append(css_score)
                                    raw_proto_set[this_box_name][proto_id]['global_position'].append(global_position)
                                else:
                                    raw_proto_set[this_box_name][proto_id] = {'points': [max_cluter],
                                                                              'outline_box': [new_box],
                                                                              'pose': [pose],
                                                                              'score': [css_score],
                                                                              'global_position': [global_position]}

            outline_infos[i]['outline_score'] = outline_score

        with open(output_pkl_path, 'wb') as f:
            pkl.dump(outline_infos, f)
        with open(output_raw_proto_path, 'wb') as f:
            pkl.dump(raw_proto_set, f)

        return outline_infos

    def limit(self, ang):
        ang = ang % (2 * np.pi)

        ang[ang > np.pi] = ang[ang > np.pi] - 2 * np.pi

        ang[ang < -np.pi] = ang[ang < -np.pi] + 2 * np.pi

        return ang


    def construct_prototypes(self,):


        seq_name, root_path, dataset_cfg = self.seq_name, self.root_path, self.dataset_cfg

        init_method_name = dataset_cfg.InitLabelGenerator

        input_raw_proto_path = os.path.join(root_path, seq_name, seq_name + '_outline_' + str(init_method_name) + '_CSS_raw_proto.pkl')

        output_proto_info_path = os.path.join(root_path, seq_name, seq_name + '_outline_'+str(init_method_name)+'_CSS_proto.pkl')

        if os.path.exists(output_proto_info_path):
            with open(output_proto_info_path, 'rb') as f:
                infos = pkl.load(f)
            return infos


        with open(input_raw_proto_path, 'rb') as f:
            raw_proto_set = pkl.load(f)

        high_quality_motion_thresh = self.dataset_cfg.RefinerConfig.HighQualityMotionThresh
        high_quality_proto_num = self.dataset_cfg.RefinerConfig.HighQualityProtoNum

        basic_proto_set = {'Vehicle': {},
                         'Pedestrian': {},
                         'Cyclist': {}}

        high_quality_proto_set = {'Vehicle': {},
                         'Pedestrian': {},
                         'Cyclist': {},}

        proto_points_set = {'Vehicle': {},
                                  'Pedestrian': {},
                                  'Cyclist': {}, }

        for cls_name in raw_proto_set.keys():
            id_list = []
            score_list = []
            points_set_list = []
            box_list = []

            for proto_id in raw_proto_set[cls_name].keys():

                all_points = raw_proto_set[cls_name][proto_id]['points']
                box_set = np.array(raw_proto_set[cls_name][proto_id]['outline_box'])
                pose_set = raw_proto_set[cls_name][proto_id]['pose']
                global_position_set = np.array(raw_proto_set[cls_name][proto_id]['global_position'])
                score_set = raw_proto_set[cls_name][proto_id]['score']
                score_mean = np.mean(score_set)

                mean_position = np.mean(global_position_set[:, 0:2], 0)
                position_dis = global_position_set[:, 0:2] - mean_position
                dis = np.linalg.norm(position_dis, axis=1)
                std = np.std(dis)

                whl_mean = np.mean(box_set[:, 3:6], 0)

                basic_proto_set[cls_name][proto_id] = whl_mean

                if std<= high_quality_motion_thresh:
                    points_set = np.concatenate(all_points, 0)
                    id_list.append(proto_id)
                    score_list.append(-score_mean)
                    points_set_list.append(points_set)
                    mean_box = copy.deepcopy(box_set[0])
                    mean_box[3:6] = whl_mean
                    pose_i = np.linalg.inv(pose_set[0])
                    new_mean_position = points_rigid_transform([mean_position], pose_i)[0:,0:3]
                    mean_box[0:3] = new_mean_position

                    max_s_arg = np.argmax(score_set)
                    angle_max = box_set[max_s_arg, 6]

                    angles = box_set[:, 6]
                    angles = self.limit(angles)
                    res = angles - angle_max
                    res = self.limit(res)
                    res = res[np.abs(res) < 1.5]
                    res = res.mean()
                    b = angle_max + res
                    mean_box[6] = b

                    box_list.append(mean_box)

                    proto_points_set[cls_name][proto_id] ={'box': mean_box, 'points': points_set, 'score': score_mean, 'move': 0}

                else:

                    arg_max = np.argmax(score_set)
                    this_points_set = all_points[arg_max]
                    max_score = score_set[arg_max]
                    max_box = box_set[arg_max]

                    pose_i = np.linalg.inv(pose_set[arg_max])

                    this_points_set[:,0:3] = points_rigid_transform(this_points_set[:,0:3], pose_set[0])[:,0:3]
                    this_points_set[:,0:3] = points_rigid_transform(this_points_set[:,0:3], pose_i)[:,0:3]

                    proto_points_set[cls_name][proto_id] = {'box': max_box, 'points': this_points_set, 'score': max_score, 'move': 1}


            if len(score_list)==0:
                continue

            arg_max_score = np.argsort(score_list)

            proto_num = min(high_quality_proto_num[cls_name], len(arg_max_score))

            arg_max_score = arg_max_score[0:proto_num]

            for list_id in arg_max_score:
                proto_id = id_list[list_id]
                proto_box = box_list[list_id]
                high_quality_proto_set[cls_name][proto_id] = {'box': proto_box}


        proto_set = {'basic_proto_set': basic_proto_set,
                     'high_quality_proto_set': high_quality_proto_set,
                     'proto_points_set': proto_points_set}

        with open(output_proto_info_path, 'wb') as f:
            pkl.dump(proto_set, f)

        return proto_set

    def refine_box_size(self):

        seq_name, root_path, dataset_cfg = self.seq_name, self.root_path, self.dataset_cfg

        init_method_name = dataset_cfg.InitLabelGenerator

        refine_method_name = dataset_cfg.LabelRefiner

        input_proto_info_path = os.path.join(root_path, seq_name, seq_name + '_outline_'+str(init_method_name)+'_CSS_proto.pkl')
        input_info_path = os.path.join(root_path, seq_name, seq_name + '_outline_' + str(init_method_name) + '_CSS.pkl')

        output_info_path = os.path.join(root_path, seq_name, seq_name + '_outline_' + str(refine_method_name) + '_resize.pkl')

        if os.path.exists(output_info_path):
            with open(output_info_path, 'rb') as f:
                infos = pkl.load(f)
            return infos


        with open(input_proto_info_path, 'rb') as f:
            proto_set = pkl.load(f)

        with open(input_info_path, 'rb') as f:
            outline_infos = pkl.load(f)

        basic_proto_set = proto_set['basic_proto_set']
        high_quality_proto_set = proto_set['high_quality_proto_set']

        high_quality_proto_whl = {'Vehicle': [[],[]],
                                  'Pedestrian': [[],[]],
                                  'Cyclist': [[],[]], }

        for cls_name in high_quality_proto_set:
            for proto_id in high_quality_proto_set[cls_name]:
                whl = high_quality_proto_set[cls_name][proto_id]['box'][3:6]
                high_quality_proto_whl[cls_name][0].append(proto_id)
                high_quality_proto_whl[cls_name][1].append(whl)


        seq_id = int(self.seq_name[8:16])
        predifined_size = self.dataset_cfg.RefinerConfig.CSSConfig.PredifinedSize

        for i in range(0, len(outline_infos)):

            info_path = str(i).zfill(4) + '.npy'
            lidar_path = os.path.join(root_path, seq_name, info_path)
            lidar_points = np.load(lidar_path)[:, 0:3]

            outline_box = outline_infos[i]['outline_box']
            outline_ids = outline_infos[i]['outline_ids']
            outline_cls = outline_infos[i]['outline_cls']
            outline_score = outline_infos[i]['outline_score']
            outline_proto_id = np.ones_like(outline_ids, dtype=np.longlong)*(-1)

            for box_id in range(len(outline_box)):
                this_box = outline_box[box_id]
                this_box_name = outline_cls[box_id]
                this_box_id = outline_ids[box_id]

                proto_id = int(str(seq_id) + str(this_box_id))

                if this_box_name not in basic_proto_set:
                    continue

                dis = np.sqrt(np.sum((lidar_points[:,0:2]-this_box[0:2])**2,-1))
                mask_low = dis<max(this_box[3],this_box[4])

                this_low_points = lidar_points[mask_low]
                if len(this_low_points)>0:
                    this_low_points = smooth_points(this_low_points)

                if len(this_low_points)>0:
                    z_min = min(this_low_points[:, 2])
                else:
                    z_min = this_box[2]-this_box[5]/2

                z_max = this_box[2]+this_box[5]/2

                h = z_max - z_min
                if h<1.3:
                    h = 1.3
                z = h/2+z_min

                if proto_id in basic_proto_set[this_box_name]:
                    fited_proto = basic_proto_set[this_box_name][proto_id]
                    outline_proto_id[box_id] = proto_id
                else:
                    this_proto_data = high_quality_proto_whl[this_box_name]
                    this_proto_ids = this_proto_data[0]
                    proto_whl = np.array(this_proto_data[1])

                    if len(this_proto_ids) == 0:
                        fited_proto = predifined_size[this_box_name]
                        fited_proto_id = -1
                        outline_proto_id[box_id] = fited_proto_id
                    else:
                        most_fit_proto_i = np.argmin(np.abs(proto_whl[:, 2] - h))
                        fited_proto = proto_whl[most_fit_proto_i]
                        fited_proto_id = this_proto_ids[most_fit_proto_i]
                        outline_proto_id[box_id] = fited_proto_id

                if this_box_name == 'Vehicle':
                    new_box = np.array([this_box[0], this_box[1], z, fited_proto[0], fited_proto[1], h, this_box[6]])
                else:
                    new_box = np.array([this_box[0], this_box[1], z, this_box[3], this_box[4], h, this_box[6]])

                if len(this_low_points) > 0:

                    valid_low_mask = this_low_points[:, 2] > z_min + 0.2
                    valid_high_mask = this_low_points[:, 2] < z_max
                    mask = valid_low_mask * valid_high_mask
                    this_low_points = this_low_points[mask]

                    non_ground_points = self.outline_estimator.remove_ground(this_low_points)

                    if len(non_ground_points) > 10:
                        clusters, labels = self.outline_estimator.clustering(non_ground_points)
                        if len(clusters) > 0:
                            max_cluter = clusters[0]
                            for clu in clusters:
                                if len(clu) > len(max_cluter):
                                    max_cluter = clu

                            if len(max_cluter) > 0:
                                css_score = self.css_estimator(max_cluter, new_box, this_box_name)
                                outline_score[box_id] = css_score

                            if this_box_name == 'Vehicle':
                                if outline_score[box_id]>self.dataset_cfg.RefinerConfig.OrienThresh:
                                    new_box = correct_orientation(max_cluter, new_box)
                                else:
                                    new_box = new_box

                                new_box = density_guided_drift(max_cluter, copy.deepcopy(new_box))

                outline_box[box_id] = new_box[:]

            outline_infos[i]['outline_box'] = outline_box
            outline_infos[i]['outline_score'] = outline_score
            outline_infos[i]['outline_proto_id'] = outline_proto_id

        with open(output_info_path, 'wb') as f:
            pkl.dump(outline_infos, f)


    def refine_box_pos(self):


        seq_name, root_path, dataset_cfg = self.seq_name, self.root_path, self.dataset_cfg


        refine_method_name = dataset_cfg.LabelRefiner

        input_info_path = os.path.join(root_path, seq_name, seq_name + '_outline_' + str(refine_method_name) + '_resize.pkl')
        output_info_path = os.path.join(root_path, seq_name, seq_name + '_outline_' + str(refine_method_name) + '.pkl')

        if os.path.exists(output_info_path):
            with open(output_info_path, 'rb') as f:
                infos = pkl.load(f)
            return infos

        with open(input_info_path, 'rb') as f:
            outline_infos = pkl.load(f)

        pos_proto = {
            'pose': {},
            'box': {},
            'proto_id': {},
            'cls': {},
            'score': {},
            'global_position': {}
        }

        for i in range(0, len(outline_infos)):

            outline_box = outline_infos[i]['outline_box']
            outline_ids = outline_infos[i]['outline_ids']
            outline_cls = outline_infos[i]['outline_cls']
            outline_score = outline_infos[i]['outline_score']
            outline_proto_id = outline_infos[i]['outline_proto_id']
            for box_i, this_box in enumerate(outline_box):
                pose = outline_infos[i]['pose']
                ob_id = outline_ids[box_i]
                ob_cls = outline_cls[box_i]
                ob_score = outline_score[box_i]
                ob_proto_id = outline_proto_id[box_i]

                global_position = points_rigid_transform(np.array([this_box[0:3]]), pose)[:, 0:3]

                if ob_id in pos_proto['proto_id'].keys():
                    pos_proto['proto_id'][ob_id][i] = ob_proto_id
                else:
                    pos_proto['proto_id'][ob_id] = {i: ob_proto_id}

                if ob_id in pos_proto['pose'].keys():
                    pos_proto['pose'][ob_id][i] = pose
                else:
                    pos_proto['pose'][ob_id] = {i: pose}

                if ob_id in pos_proto['box'].keys():
                    pos_proto['box'][ob_id][i] = this_box
                else:
                    pos_proto['box'][ob_id] = {i: this_box}

                if ob_id in pos_proto['cls'].keys():
                    pos_proto['cls'][ob_id][i] = ob_cls
                else:
                    pos_proto['cls'][ob_id] = {i: ob_cls}

                if ob_id in pos_proto['score'].keys():
                    pos_proto['score'][ob_id][i] = ob_score
                else:
                    pos_proto['score'][ob_id] = {i: ob_score}

                if ob_id in pos_proto['global_position'].keys():
                    pos_proto['global_position'][ob_id][i] = global_position[0]
                else:
                    pos_proto['global_position'][ob_id] = {i: global_position[0]}

        new_pos_proto_static = {
            'pose': {},
            'box': {},
            'cls': {},
            'proto_id': {},
            'score': {}
        }

        new_pos_proto_dynamic = {
            'pose': {},
            'box': {},
            'cls': {},
            'proto_id': {},
            'score': {}
        }

        for ob_id in pos_proto['box'].keys():

            all_score = np.array(list(pos_proto['score'][ob_id].values()))
            all_box = np.array(list(pos_proto['box'][ob_id].values()))
            all_cls = np.array(list(pos_proto['cls'][ob_id].values()))
            all_proto_id = np.array(list(pos_proto['proto_id'][ob_id].values()))
            all_pose = np.array(list(pos_proto['pose'][ob_id].values()))
            all_position = np.array(list(pos_proto['global_position'][ob_id].values()))

            mean_position = np.mean(all_position[:, 0:2], 0)
            position_dis = all_position[:, 0:2] - mean_position
            dis = np.linalg.norm(position_dis, axis=1)
            std = np.std(dis)

            argmax_score = np.argmax(all_score)

            best_pos = all_pose[argmax_score]
            best_box = all_box[argmax_score]
            best_cls = all_cls[argmax_score]
            best_score = all_score[argmax_score]
            best_proto_id = all_proto_id[argmax_score]

            if std < self.dataset_cfg.RefinerConfig.StaticThresh:

                new_pos_proto_static['pose'][ob_id] = best_pos
                new_pos_proto_static['box'][ob_id] = best_box
                new_pos_proto_static['cls'][ob_id] = best_cls
                new_pos_proto_static['score'][ob_id] = best_score
                new_pos_proto_static['proto_id'][ob_id] = best_proto_id
            else:
                if ob_id not in new_pos_proto_dynamic['box'].keys():
                    new_pos_proto_dynamic['box'][ob_id] = {}
                    new_pos_proto_dynamic['score'][ob_id] = {}
                    new_pos_proto_dynamic['cls'][ob_id] = {}
                    new_pos_proto_dynamic['proto_id'][ob_id] = {}
                for frame_id in pos_proto['box'][ob_id].keys():
                    new_pos_proto_dynamic['score'][ob_id][frame_id] = best_score
                    this_box = copy.deepcopy(pos_proto['box'][ob_id][frame_id])
                    this_box[3:6] = best_box[3:6]
                    new_pos_proto_dynamic['box'][ob_id][frame_id] = this_box
                    new_pos_proto_dynamic['cls'][ob_id][frame_id] = best_cls
                    new_pos_proto_dynamic['proto_id'][ob_id][frame_id] = best_proto_id
                    this_pose = pos_proto['pose'][ob_id][frame_id]
                    position_left = []
                    position_right = []

                    win_size = 10

                    for left_i in range(frame_id - win_size + 1, frame_id + 1):
                        if left_i in pos_proto['global_position'][ob_id].keys():
                            position_left.append(pos_proto['global_position'][ob_id][left_i])

                    for right_i in range(frame_id, frame_id + win_size):
                        if right_i in pos_proto['global_position'][ob_id].keys():
                            position_right.append(pos_proto['global_position'][ob_id][right_i])

                    position_left = np.array(position_left)
                    position_right = np.array(position_right)

                    position_left = np.mean(position_left[:, 0:2], 0)
                    position_right = np.mean(position_right[:, 0:2], 0)

                    angle_vec = position_right - position_left

                    this_dis = np.linalg.norm(angle_vec)

                    if this_dis > 1:
                        global_angle = angle_from_vector(angle_vec[0], angle_vec[1])

                        pos_inv = np.linalg.inv(this_pose)

                        angle_off_from_pose = get_registration_angle(pos_inv)

                        real_angle = global_angle + angle_off_from_pose

                        new_pos_proto_dynamic['box'][ob_id][frame_id][6] = real_angle

        for i in range(0, len(outline_infos)):


            outline_box = outline_infos[i]['outline_box']
            outline_ids = outline_infos[i]['outline_ids']

            for box_i, this_box in enumerate(outline_box):
                ob_id = outline_ids[box_i]


                if ob_id in new_pos_proto_static['box'].keys():

                    proto_box = new_pos_proto_static['box'][ob_id]
                    proto_pos = new_pos_proto_static['pose'][ob_id]
                    this_pose = outline_infos[i]['pose']
                    propo_cls = new_pos_proto_static['cls'][ob_id]
                    proto_score = new_pos_proto_static['score'][ob_id]
                    proto_id = new_pos_proto_static['proto_id'][ob_id]

                    new_box = box_rigid_transform(proto_box, proto_pos, this_pose)

                    outline_infos[i]['outline_box'][box_i] = new_box[:]
                    outline_infos[i]['outline_cls'][box_i] = propo_cls
                    if propo_cls in self.dataset_cfg.RefinerConfig.BasicProtoScoreThresh:
                        if proto_score > self.dataset_cfg.RefinerConfig.BasicProtoScoreThresh[propo_cls]:
                            outline_infos[i]['outline_score'][box_i] = proto_score
                    outline_infos[i]['outline_proto_id'][box_i] = proto_id

        with open(output_info_path, 'wb') as f:
            pkl.dump(outline_infos, f)

        return outline_infos

    def __call__(self, ):
        infos = self.compute_css_score_and_raw_proto()
        infos = self.construct_prototypes()
        infos = self.refine_box_size()
        infos = self.refine_box_pos()

        return infos
import os
import pickle as pkl
from pcdet.unsupervised_core.outline_utils import KL_entropy_score, TrackSmooth
import numpy as np
from pcdet.unsupervised_core.rotate_iou_cpu_eval import rotate_iou_cpu_one
import copy

def NMS(boxes, scores):

    if len(boxes)<=1:
        return boxes,scores
    selected = []
    selected_s = []
    index = np.argsort(-scores)
    boxes = boxes[index]
    scores = scores[index]

    selected.append(boxes[0])
    selected_s.append(scores[0])

    for i, box in enumerate(boxes):
        this_iou = 0
        s = scores[i]
        for se_box in selected:
            try:
                this_iou,_ = rotate_iou_cpu_one((box,se_box))
            except:
                this_iou = 0.1
            dis = np.linalg.norm(box[:3]-se_box[:3]) < max(box[3]/2,se_box[3]/2)
            if this_iou>=0.01 or dis:
                break
        if this_iou>=0.01 or dis:
            continue

        selected.append(box)
        selected_s.append(s)

    return np.array(selected), np.array(selected_s)


class PseudoLabel():
    def __init__(self, seq_name, root_path, config):
        self.seq_name = seq_name
        self.root_path = root_path
        self.dataset_cfg = config

    def compute_cls(self, boxes):

        size_temp = self.dataset_cfg.GeneratorConfig.PredifinedSize

        cls_names = ['Vehicle', 'Pedestrian', 'Cyclist']

        cls_final = []

        for box in boxes:

            cls_score = []
            cur_whl = box[3:6]

            cur_whl = cur_whl / cur_whl.sum()

            for cls in cls_names:
                cur_temp = np.array(size_temp[cls])

                cur_temp = cur_temp/cur_temp.sum()
                cls_score.append(KL_entropy_score(cur_whl, cur_temp))

            this_cls = cls_names[np.argmax(cls_score)]
            cls_final.append(this_cls)

        return np.array(cls_final)


    def to_pesudo_label(self):

        seq_name, root_path, dataset_cfg = self.seq_name, self.root_path, self.dataset_cfg

        method_name = dataset_cfg.InitLabelGenerator

        output_pkl_path = os.path.join(root_path, seq_name, seq_name + '_outline_'+str(method_name)+'.pkl')

        # if os.path.exists(output_pkl_path):
        #     with open(output_pkl_path, 'rb') as f:
        #         infos = pkl.load(f)
        #     return infos

        with open(os.path.join(self.dataset_cfg.MERGE_CONFIG.DET_PATH, 'result_merged.pkl'), 'rb') as f:
            merged_infos = pkl.load(f)

        input_pkl_path = os.path.join(root_path, seq_name, seq_name + '.pkl')

        with open(input_pkl_path, 'rb') as f:
            save_infos = pkl.load(f)

        pred_seq_annos = []
        pred_seq_scores = []
        for index in range(len(merged_infos)):

            det_names = merged_infos[index]['name']
            det_scores = merged_infos[index]['score']
            det_boxes = merged_infos[index]['boxes_lidar']
            seq_id = merged_infos[index]['seq_id']

            if seq_id == seq_name:
                mask = np.zeros_like(det_scores)

                for cls in self.dataset_cfg.GeneratorConfig.moving_score_thresh:
                    score_thresh_d = self.dataset_cfg.GeneratorConfig.moving_score_thresh[cls]
                    valid_mask = det_scores>score_thresh_d
                    valid_cls = det_names==cls
                    this_mask = valid_mask*valid_cls
                    mask+=this_mask

                det_boxes = det_boxes[mask.astype(bool)]
                scores = det_scores[mask.astype(bool)]

                det_boxes, scores = NMS(det_boxes, scores)
                pred_seq_annos.append(det_boxes)
                pred_seq_scores.append(scores)

        all_pose = []
        for i, this_info in enumerate(pred_seq_annos):
            all_pose.append(save_infos[i]['pose'])

        tracker = TrackSmooth(self.dataset_cfg.GeneratorConfig)
        tracker.tracking(pred_seq_annos, all_pose, pred_seq_scores)

        for i in range(0, len(save_infos)):

            objs, ids, cls, dif, std, speed, score = tracker.get_current_frame_objects_and_cls_mv(i)

            new_cls = self.compute_cls(objs)

            save_infos[i]['outline_box'] = objs
            save_infos[i]['outline_ids'] = ids
            save_infos[i]['outline_cls'] = new_cls
            save_infos[i]['outline_dif'] = dif
            save_infos[i]['outline_std'] = std
            save_infos[i]['outline_speed'] = speed
            save_infos[i]['outline_score'] = score


        for i in range(0, len(save_infos)):

            objs = save_infos[i]['outline_box']
            ids = save_infos[i]['outline_ids']
            det_names = save_infos[i]['outline_cls']
            dif = save_infos[i]['outline_dif']
            std = save_infos[i]['outline_std']
            speed = save_infos[i]['outline_speed']
            score = save_infos[i]['outline_score']


            cyclist_size = np.array(self.dataset_cfg.GeneratorConfig.PredifinedSize['Cyclist'])

            mask_ped = det_names=='Pedestrian'
            mask_speed = speed>self.dataset_cfg.GeneratorConfig.cyclist_speed

            mask = mask_ped*mask_speed

            if mask.sum() > 0:
                det_names[mask] = 'Cyclist'

            mask_cyc = det_names == 'Cyclist'

            new_ob = copy.deepcopy(objs[mask_cyc])

            if len(new_ob) > 0:
                new_ob[:, 3:6] = cyclist_size
                objs[mask_cyc] = new_ob



            mask_dynamic = np.zeros_like(std)
            mask_static = np.zeros_like(std)

            for cls in self.dataset_cfg.GeneratorConfig.moving_thresh:
                thresh = self.dataset_cfg.GeneratorConfig.moving_thresh[cls]
                score_thresh_d = self.dataset_cfg.GeneratorConfig.moving_score_thresh[cls]
                score_thresh_s = self.dataset_cfg.GeneratorConfig.static_score_thresh[cls]

                valid_mask_d = std >= thresh
                valid_mask_s = std < thresh
                valid_cls = det_names == cls

                valid_score_d = score > score_thresh_d
                valid_score_s = score > score_thresh_s

                this_mask_d = valid_mask_d * valid_cls * valid_score_d
                this_mask_s = valid_mask_s * valid_cls * valid_score_s

                mask_dynamic += this_mask_d
                mask_static += this_mask_s

            mask_dynamic = mask_dynamic.astype(bool)
            mask_static = mask_static.astype(bool)


            objs = np.concatenate([objs[mask_dynamic], objs[mask_static]])
            ids = np.concatenate([ids[mask_dynamic], ids[mask_static]])
            det_names = np.concatenate([det_names[mask_dynamic], det_names[mask_static]])
            dif = np.concatenate([dif[mask_dynamic], dif[mask_static]])
            speed = np.concatenate([speed[mask_dynamic], speed[mask_static]])
            score = np.concatenate([score[mask_dynamic], score[mask_static]])
            std = np.concatenate([std[mask_dynamic], std[mask_static]])

            save_infos[i]['outline_box'] = objs
            save_infos[i]['outline_ids'] = ids
            save_infos[i]['outline_cls'] = det_names
            save_infos[i]['outline_dif'] = dif
            save_infos[i]['outline_speed'] = speed
            save_infos[i]['outline_score'] = score
            save_infos[i]['outline_std'] = std

        with open(output_pkl_path, 'wb') as f:
            pkl.dump(save_infos, f)

        return save_infos

    def __call__(self,):
        return self.to_pesudo_label()
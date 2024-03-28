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


class PseudoLabelS():
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

    def filtering_based_on_gt(self,det_scores,det_boxes,gt_boxes,gt_name):

        new_name = []
        new_score = []
        new_box = []
        if len(det_boxes)<=0 or len(gt_name)<=0:
            return np.array(new_name), np.array(new_score), np.array(new_box)

        for i in range(len(det_boxes)):
            box = det_boxes[i]
            s = det_scores[i]

            dis = np.linalg.norm(gt_boxes[:,0:3]-box[0:3], axis=-1)

            if len(dis)<=0 or len(gt_name)<=0:
                continue

            cls = gt_name[np.argmin(dis)]

            if np.min(dis)<1:
                new_name.append(cls)
                new_score.append(s)
                new_box.append(box)

        return np.array(new_name), np.array(new_score), np.array(new_box)



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

                det_boxes = det_boxes
                scores = det_scores

                det_boxes, scores = NMS(det_boxes, scores)
                pred_seq_annos.append(det_boxes)
                pred_seq_scores.append(scores)


        for i in range(0, len(save_infos)):


            gt_boxes = save_infos[i]['annos']['gt_boxes_lidar']
            gt_names = save_infos[i]['annos']['name']

            det_boxes, det_scores = pred_seq_annos[i], pred_seq_scores[i]

            det_names, det_scores, det_boxes = self.filtering_based_on_gt(det_scores, det_boxes, gt_boxes, gt_names)

            save_infos[i]['outline_box'] = det_boxes
            save_infos[i]['outline_ids'] = det_scores
            save_infos[i]['outline_cls'] = det_names
            save_infos[i]['outline_dif'] = det_scores
            save_infos[i]['outline_std'] = det_scores
            save_infos[i]['outline_speed'] = det_scores
            save_infos[i]['outline_score'] = det_scores


        with open(output_pkl_path, 'wb') as f:
            pkl.dump(save_infos, f)

        return save_infos

    def __call__(self,):
        return self.to_pesudo_label()
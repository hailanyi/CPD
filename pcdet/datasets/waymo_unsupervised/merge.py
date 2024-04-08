from cpd.ops.iou3d_nms import iou3d_nms_utils
from cpd.config import *
import os
import pickle
import numpy as np
import torch
from tqdm import trange
from cpd.utils import common_utils


def read_pickles(path, cfg):

    all_pickles = []
    for root, dirs, files in os.walk(path, topdown=True):
        for name in files:
            flag = True

            if name[-3:] == 'pkl' and root[-5:]=='train' in root and ('result_merged.pkl' != name) and flag:
                cur_path = os.path.join(root, name)
                print(cur_path)
                with open(cur_path, 'rb') as f:
                    infos = pickle.load(f)
                    all_pickles.append(infos)

    #all_test_iters = glob.glob(os.path.join(path, 'eval_with_train_*'))
    #all_test_iters.sort(key=os.path.getmtime)

    #for cur_iter_path in all_test_iters:
    #    all_epochs = glob.glob(os.path.join(cur_iter_path, 'epoch_*'))
    #    for cur_epoch_path in all_epochs:
    #        cur_pickle_path = Path(cur_epoch_path)/split/'result.pkl'
    #        print(cur_pickle_path)

    return all_pickles


class DetectionsMerger():
    def __init__(self, config):
        self.det_path = config.MERGE_CONFIG.DET_PATH
        self.config = config
        self.config = self.config.MERGE_CONFIG
        all_pickle_file_path = os.listdir(self.det_path)

        self.all_infos=read_pickles(self.det_path, self.config)

    def prepare_detections(self,index):
        det_boxes = []
        det_scores = []
        det_names = []

        for infos in self.all_infos:
            det_boxes.append(infos[index]['boxes_lidar'])
            det_scores.append(infos[index]['score'])
            det_names.append(infos[index]['name'])


        det_boxes = np.concatenate(det_boxes)
        det_scores = np.concatenate(det_scores)
        det_names = np.concatenate(det_names)

        return det_names,det_scores,det_boxes

    def merge_by_NMS(self,det_names, det_scores, det_boxes):

        if len(det_names)==0:
            return det_names, det_scores, det_boxes

        iou_thresh = self.config.NMS_IOU
        det_boxes = torch.from_numpy(det_boxes[:,0:7].astype(np.float32)).cuda()
        det_scores = torch.from_numpy(det_scores.astype(np.float32)).cuda()

        orders = iou3d_nms_utils.nms_gpu(det_boxes,det_scores,iou_thresh)[0]

        det_boxes = det_boxes.cpu().numpy()
        det_scores = det_scores.cpu().numpy()

        orders = orders.cpu().numpy()
        return det_names[orders],det_scores[orders],det_boxes[orders]

    def limit(self, ang):
        ang = ang % (2 * np.pi)

        ang[ang > np.pi] = ang[ang > np.pi] - 2 * np.pi

        ang[ang < -np.pi] = ang[ang < -np.pi] + 2 * np.pi

        return ang


    def compute_WBF(self, det_names, det_scores, det_boxes, iou_thresh, type = 'mean'):
        if len(det_names) == 0:
            return det_names, det_scores, det_boxes
        cluster_id = -1
        cluster_box_dict = {}
        cluster_score_dict = {}

        cluster_merged_dict = {}
        '''
        det_boxes[:, 6] = common_utils.limit_period(
            det_boxes[:, 6], offset=0.5, period=2 * np.pi
        )
        '''
        det_boxes[:, 6] = self.limit(det_boxes[:, 6])

        for i, box in enumerate(det_boxes):

            score = det_scores[i]
            if i == 0:
                cluster_id+=1
                cluster_box_dict[cluster_id] = [box]
                cluster_score_dict[cluster_id] = [score]
                cluster_merged_dict[cluster_id] = box
                continue

            valid_clusters = []
            keys = list(cluster_merged_dict)
            keys.sort()
            for key in keys:
                valid_clusters.append(cluster_merged_dict[key])

            valid_clusters = np.array(valid_clusters).reshape((-1,7))
            ious = iou3d_nms_utils.boxes_bev_iou_cpu(np.array([box[:7]]), valid_clusters)


            argmax = np.argmax(ious,-1)[0]
            max_iou = np.max(ious,-1)[0]

            if max_iou>iou_thresh:
                cluster_box_dict[argmax].append(box)
                cluster_score_dict[argmax].append(score)
            else:
                cluster_id += 1
                cluster_box_dict[cluster_id] = [box]
                cluster_score_dict[cluster_id] = [score]
                cluster_merged_dict[cluster_id] = box

        out_boxes = []
        out_scores = []
        for i in cluster_merged_dict.keys():
            if type == 'mean':
                score_sum = 0
                box_sum = np.zeros(shape=(7,))

                angles = []

                for j, sub_score in enumerate(cluster_score_dict[i]):
                    box_sum += cluster_box_dict[i][j]
                    score_sum += sub_score
                    angles.append(cluster_box_dict[i][j][6])
                box_sum /= len(cluster_score_dict[i])
                score_sum /= len(cluster_score_dict[i])

                cluster_merged_dict[i][:6] = box_sum[:6]

                angles = np.array(angles)
                angles = self.limit(angles)
                res = angles - cluster_merged_dict[i][6]
                res = self.limit(res)
                res = res[np.abs(res)<1.5]
                res = res.mean()
                b = cluster_merged_dict[i][6] + res
                cluster_merged_dict[i][6] = b

                out_scores.append(score_sum)
                out_boxes.append(cluster_merged_dict[i])
            elif type == 'max':
                out_scores.append(np.max(cluster_score_dict[i]))
                out_boxes.append(cluster_merged_dict[i])


        out_boxes = np.array(out_boxes)
        out_scores = np.array(out_scores)
        out_names = det_names[0:out_boxes.shape[0]]

        return out_names,out_scores,out_boxes



    def merge_by_WBF(self,det_names, det_scores, det_boxes):
        wbf_iou = self.config.WBF_IOUS
        wbf_pre_score = self.config.WBF_PRE_SCORES

        classes = ['Vehicle', 'Pedestrian', 'Cyclist']

        all_cls_names = []
        all_cls_scores = []
        all_cls_boxes = []

        for i,cls in enumerate(classes):
            name_mask = det_names == cls
            sub_names = det_names[name_mask]
            sub_scores = det_scores[name_mask]
            sub_boxes = det_boxes[name_mask]
            thresh_mask = sub_scores>=wbf_pre_score[i]
            sub_names = sub_names[thresh_mask]
            sub_scores = sub_scores[thresh_mask]
            sub_boxes = sub_boxes[thresh_mask]

            arg_sorts = np.argsort(-sub_scores)
            sub_names = sub_names[arg_sorts]
            sub_scores = sub_scores[arg_sorts]
            sub_boxes = sub_boxes[arg_sorts]

            iou_thresh = wbf_iou[i]

            type = self.config.MERGE_TYPE[i]

            if type == 'WBF-mean':
                sub_names, sub_scores, sub_boxes = self.compute_WBF(sub_names, sub_scores, sub_boxes, iou_thresh)
            elif type == 'WBF-max':
                sub_names, sub_scores, sub_boxes = self.compute_WBF(sub_names, sub_scores, sub_boxes, iou_thresh, 'max')
            elif type == 'NMS':
                sub_names, sub_scores, sub_boxes = self.merge_by_NMS(sub_names, sub_scores, sub_boxes)

            all_cls_names.append(sub_names)
            all_cls_scores.append(sub_scores)
            all_cls_boxes.append(sub_boxes)

        all_cls_names = np.concatenate(all_cls_names)
        all_cls_scores = np.concatenate(all_cls_scores)
        all_cls_boxes = np.concatenate(all_cls_boxes)

        return all_cls_names,all_cls_scores,all_cls_boxes


    def merge(self):
        out_path = os.path.join(self.det_path, 'result_merged.pkl')
        out_infos = self.all_infos[0]

        for index in trange(len(out_infos)):
            det_names, det_scores, det_boxes = self.prepare_detections(index)

            det_names, det_scores, det_boxes = self.merge_by_WBF(det_names, det_scores, det_boxes)

            out_infos[index]['name'] = det_names
            out_infos[index]['score'] = det_scores
            out_infos[index]['boxes_lidar'] = det_boxes

        with open(out_path, 'wb') as f:
            pickle.dump(out_infos, f)

    def to_pseudo_label(self, preds, root_path):

        all_pred_seq_annos = {}
        print('preds are converted to labels! ')
        for index in trange(len(preds)):

            det_names = preds[index]['name']
            det_scores = preds[index]['score']
            det_boxes = preds[index]['boxes_lidar']
            seq_id = preds[index]['seq_id']

            if seq_id in all_pred_seq_annos:
                all_pred_seq_annos[seq_id].append([det_names, det_scores, det_boxes])
            else:
                all_pred_seq_annos[seq_id] = [[det_names, det_scores, det_boxes]]
        this_c = 0
        for seq_id in all_pred_seq_annos:
            print(this_c, '\\', len(all_pred_seq_annos))
            pkl_info_in_path = os.path.join(root_path, seq_id, seq_id+'.pkl')
            pkl_info_out_path = os.path.join(root_path, seq_id, seq_id + '_SI.pkl')

            with open(pkl_info_in_path, 'rb') as f:
                save_infos = pickle.load(f)

            for i, this_info in enumerate(save_infos):
                det_names = all_pred_seq_annos[seq_id][i][0]
                det_scores = all_pred_seq_annos[seq_id][i][1]
                det_boxes = all_pred_seq_annos[seq_id][i][2]
                save_infos[i]['outline_cls'] = det_names
                save_infos[i]['outline_score'] = det_scores
                save_infos[i]['outline_box'] = det_boxes
            this_c+=1
            with open(pkl_info_out_path, 'wb') as f:
                pickle.dump(save_infos, f)




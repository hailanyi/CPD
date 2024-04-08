from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.config import *
import os
import pickle
import numpy as np
import torch
from tqdm import trange
from pcdet.utils import common_utils
from skimage import io
from pathlib import Path
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti



class DetectionsMerger():
    def __init__(self,config_path,det_path,split = 'val'):
        self.det_path = det_path
        self.data_config = cfg_from_yaml_file(config_path,cfg)
        self.config = self.data_config.TEST_AUGMENTOR.MERGE_CONFIG
        all_pickle_file_path = os.listdir(self.det_path)
        self.all_infos = {}
        self.split = split

        for file_name in all_pickle_file_path:
            if file_name[-3:]!='pkl' or file_name == 'result_merged_tracking.pkl':
                continue
            info_path = os.path.join(self.det_path,file_name)
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)

                self.all_infos[file_name]=infos

        self.root_split_path = Path(self.data_config.DATA_PATH) / ('training' if split != 'test' else 'testing')
 
        self.seq_ids = []

        for seq in os.listdir(self.root_split_path / "velodyne"):
            if self.split == 'train' and int(seq) not in dataset_cfg.VAL_SEQ:
                self.seq_ids.append(int(seq))
            if self.split == 'val' and int(seq) in dataset_cfg.VAL_SEQ:
                self.seq_ids.append(int(seq))
            if self.split == 'trainval' or self.split == 'test':
                self.seq_ids.append(int(seq))

        self.seq_ids.sort()
        self.sample_id_list = self.get_sample_id_list()


        self.image_info = []
        self.calib_info = []

        for id in self.sample_id_list:
            self.image_info.append(np.array([374,1241]))
            self.calib_info.append(self.get_calib(id))

    def get_sample_id_list(self):

        sample_id_list=[]

        for seq_id in self.seq_ids:
            point_seq_path = self.root_split_path / 'velodyne' / str(seq_id).zfill(4)

            frame_ids=os.listdir(point_seq_path)
            frame_ids.sort()

            for name in frame_ids:
                sample_id_list.append((seq_id,int(name[:6])))

        return sample_id_list

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ( str(idx[0]).zfill(4)+".txt")
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def prepare_detections(self,index):
        det_boxes = []
        det_scores = []
        det_names = []

        for key in self.all_infos.keys():
            infos = self.all_infos[key]
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

    def compute_WBF(self, det_names, det_scores, det_boxes, iou_thresh,type = 'mean'):
        if len(det_names) == 0:
            return det_names, det_scores, det_boxes
        cluster_id = -1
        cluster_box_dict = {}
        cluster_score_dict = {}

        cluster_merged_dict = {}
        det_boxes[:, 6] = common_utils.limit_period(
            det_boxes[:, 6], offset=0.5, period=2 * np.pi
        )
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
            ious = iou3d_nms_utils.boxes_bev_iou_cpu(np.array([box[:7]]),valid_clusters)

            argmax = np.argmax(ious,-1)[0]
            max_iou = np.max(ious,-1)[0]

            if max_iou>iou_thresh:
                cluster_box_dict[argmax].append(box)
                cluster_score_dict[argmax].append(score)
                if type == 'mean':
                    merged_box = np.zeros(shape=(7,))
                    score_sum = 0

                    for j, sub_box in enumerate(cluster_box_dict[argmax]):
                        sub_score = cluster_score_dict[argmax][j]
                        merged_box += sub_box*sub_score
                        score_sum += sub_score

                    merged_box/=score_sum
                    cluster_merged_dict[argmax][0:6] = merged_box[0:6]

            else:
                cluster_id += 1
                cluster_box_dict[cluster_id] = [box]
                cluster_score_dict[cluster_id] = [score]
                cluster_merged_dict[cluster_id] = box
        out_boxes = []
        out_scores = []
        for i in cluster_merged_dict.keys():
            out_boxes.append(cluster_merged_dict[i])
            if type == 'mean':
                score_sum = 0
                for j, sub_score in enumerate(cluster_score_dict[i]):

                    score_sum += sub_score
                score_sum /= len(cluster_score_dict[i])
                out_scores.append(score_sum)
            elif type == 'max':
                out_scores.append(np.max(cluster_score_dict[i]) )
        out_boxes = np.array(out_boxes)
        out_scores = np.array(out_scores)
        out_names = det_names[0:out_boxes.shape[0]]

        return out_names,out_scores,out_boxes


    def merge_by_WBF(self,det_names, det_scores, det_boxes):
        wbf_iou = self.config.WBF_IOUS
        wbf_pre_score = self.config.WBF_PRE_SCORES

        classes = ['Car', 'Pedestrian', 'Cyclist']

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
        out_path = os.path.join(self.det_path,'result_merged_tracking.pkl')
        all_pickle_file_path = os.listdir(self.det_path)
        for file_name in all_pickle_file_path:
            if file_name[-3:]=='pkl':
                with open(os.path.join(self.det_path, file_name), 'rb') as f:
                    out_infos = pickle.load(f)
                break

        for index in trange(len(out_infos)):
            det_names, det_scores, det_boxes = self.prepare_detections(index)

            det_names, det_scores, det_boxes = self.merge_by_WBF(det_names, det_scores, det_boxes)

            out_infos[index]['name'] = det_names
            out_infos[index]['score'] = det_scores
            out_infos[index]['boxes_lidar'] = det_boxes

            #det_boxes[:,2]+=det_boxes[:,5]/2

            calib = self.calib_info[index]
            image_shape = self.image_info[index]
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(det_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )
            out_infos[index]['alpha'] = -np.arctan2(-det_boxes[:, 1], det_boxes[:, 0]) + pred_boxes_camera[:, 6]
            out_infos[index]['bbox'] = pred_boxes_img
            out_infos[index]['dimensions'] = pred_boxes_camera[:, 3:6]
            out_infos[index]['location'] = pred_boxes_camera[:, 0:3]
            out_infos[index]['rotation_y'] = pred_boxes_camera[:, 6]



        with open(out_path, 'wb') as f:
            pickle.dump(out_infos, f)


if __name__ == '__main__':
    config_path = '/home/asc01/projects/OpenProjects/OpenPCDet/tools/cfgs/multi_frame/dataset_configs/kitti_tracking_dataset_trainval.yaml'
    det_path = '/home/asc01/projects/OpenProjects/OpenPCDet/output/multi_frame/kitti_models/voxel_rcnn_cascade_part/default/eval/eval_with_train_n_0/alls'

    merger = DetectionsMerger(config_path,det_path, 'trainval')
    merger.merge()
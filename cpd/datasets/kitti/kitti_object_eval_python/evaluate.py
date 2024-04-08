import time

import fire

import cpd.datasets.kitti.kitti_object_eval_python.kitti_common as kitti
from cpd.datasets.kitti.kitti_object_eval_python.eval import get_coco_eval_result, get_official_eval_result
import pickle

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate(label_path,
             result_path,
             label_split_file,
             current_class=[0,1,2],
             coco=False,
             score_thresh=-1):
    dt_annos = pickle.load(open(result_path,'rb'))#kitti.get_label_annos(result_path)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    if coco:
        return get_coco_eval_result(gt_annos, dt_annos, current_class)
    else:
        return get_official_eval_result(gt_annos, dt_annos, current_class)

def evaluate_dis(label_path,
             result_path,
             label_split_file,
             current_class=[0,1,2],
             coco=False,
             min_dis = 0,
             max_dis = 100):
    dt_annos = pickle.load(open(result_path,'rb'))
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)

    gt_annos = kitti.filter_gt_annos_dis(gt_annos,min_dis,max_dis)
    dt_annos = kitti.filter_det_annos_dis(dt_annos, min_dis, max_dis)

    if coco:
        return get_coco_eval_result(gt_annos, dt_annos, current_class)
    else:
        return get_official_eval_result(gt_annos, dt_annos, current_class)



def eval():
    label_path = '/home/asc01/data/kitti/detection/training/label_2'
    #det_path = 'result_merged_val.pkl'
    det_path = 'result.pkl'
    print(det_path)
    file = '/home/asc01/data/kitti/detection/ImageSets/val.txt'
    results = evaluate(label_path,det_path,file)
    print(results[0])

def eval_dis(min=60, max=70):
    label_path = '/home/asc01/data/kitti/detection/training/label_2'
    det_path = 'result_merged_val.pkl'
    file = '/home/asc01/data/kitti/detection/ImageSets/val.txt'
    results = evaluate_dis(label_path, det_path, file, min_dis=min, max_dis=max)
    print(results[0])

if __name__ == '__main__':
    eval()
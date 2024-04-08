from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pickle as pkl
import cv2
import tensorflow as tf
from waymo_open_dataset.camera.ops import py_camera_model_ops


def points_rigid_transform(cloud, pose):
    if cloud.shape[0] == 0:
        return cloud
    mat = np.ones(shape=(cloud.shape[0], 4), dtype=np.float32)
    pose_mat = np.mat(pose)
    mat[:, 0:3] = cloud[:, 0:3]
    mat = np.mat(mat)
    transformed_mat = pose_mat * mat.T
    T = np.array(transformed_mat.T, dtype=np.float32)
    return T[:, 0:3]

root = '/home/asc01/data/waymo/perception/waymo_processed_data_train_val_test'
seq_name = 'segment-10017090168044687777_6380_000_6400_000_with_camera_labels'
pkl_path = os.path.join(root, seq_name, seq_name+'.pkl')

with open(pkl_path, 'rb') as f:
    info = pkl.load(f)

for i, info_i in enumerate(info):

    image_info = info_i['image']

    lidar_path = os.path.join(root, seq_name, str(i).zfill(4)+'.npy')

    box = info_i['annos']['gt_boxes_lidar']

    pose = info_i['pose']

    lidar_points = np.load(lidar_path)

    lidar_points = points_rigid_transform(lidar_points, pose)

    lidar_points = tf.constant(lidar_points, dtype=tf.float64)

    for j in range(5):

        if j!=2 or i!=10:
            continue

        intrinsic = image_info['intrinsic_waymo_%d'%j]
        extrinsic = image_info['extrinsic_%d'%j]
        height = image_info['image_shape_%d'%j][0]
        width = image_info['image_shape_%d' % j][1]
        rolling_shutter_direction = image_info['rolling_shutter_direction_%d' % j]

        metadata = tf.constant([
            width, height,
            rolling_shutter_direction
        ], dtype=tf.int32)

        camera_image_metadata = tf.constant(image_info['camera_image_metadata_%d' % j], dtype=tf.float64)


        extrinsic = tf.constant(extrinsic, dtype=tf.float64)
        intrinsic = tf.constant(intrinsic, dtype=tf.float64)

        image = cv2.imread(os.path.join(root, seq_name, 'image', str(i).zfill(4), str(j)+'.jpg'))
        height,width,_ = image.shape

        image_points_t = py_camera_model_ops.world_to_image(
            extrinsic, intrinsic, metadata, camera_image_metadata, lidar_points)


        x = np.clip(image_points_t[:, 0], 2, width - 2)
        y = np.clip(image_points_t[:, 1], 2, height - 2)

        x = x.astype(np.int)
        y = y.astype(np.int)

        image[y, x] = (0, 0, 255)

        x2 = x + 1
        image[y, x2] = (0, 0, 255)
        y2 = y + 1
        image[y2, x] = (0, 0, 255)
        image[y2, x2] = (0, 0, 255)

        cv2.imwrite('jpg.jpg', image)

        print('suc')
        input()
        break

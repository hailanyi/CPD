# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.


import os
import pickle
import numpy as np
import sys
sys.path.append(".../")
from cpd.utils import common_utils
# from ...utils import common_utils
import tensorflow as tf
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2
import cv2
import copy

try:
    tf.enable_eager_execution()
except:
    pass


#gpus = tf.config.experimental.list_physical_devices('GPU')#
#tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])



WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']


def generate_labels(frame):
    obj_name, difficulty, dimensions, locations, heading_angles = [], [], [], [], []
    tracking_difficulty, speeds, accelerations, obj_ids = [], [], [], []
    num_points_in_gt = []

    speed_x=[]
    speed_y=[]
    acceleration_x = []
    acceleration_y = []

    laser_labels = frame.laser_labels
    for i in range(len(laser_labels)):
        box = laser_labels[i].box
        class_ind = laser_labels[i].type
        loc = [box.center_x, box.center_y, box.center_z]
        heading_angles.append(box.heading)
        obj_name.append(WAYMO_CLASSES[class_ind])
        difficulty.append(laser_labels[i].detection_difficulty_level)
        tracking_difficulty.append(laser_labels[i].tracking_difficulty_level)
        dimensions.append([box.length, box.width, box.height])  # lwh in unified coordinate of OpenPCDet
        locations.append(loc)
        obj_ids.append(laser_labels[i].id)
        num_points_in_gt.append(laser_labels[i].num_lidar_points_in_box)

        speed_x.append(laser_labels[i].metadata.speed_x)
        speed_y.append(laser_labels[i].metadata.speed_y)
        acceleration_x.append(laser_labels[i].metadata.accel_x)
        acceleration_y.append(laser_labels[i].metadata.accel_y)

    annotations = {}
    annotations['name'] = np.array(obj_name)
    annotations['difficulty'] = np.array(difficulty)
    annotations['dimensions'] = np.array(dimensions)
    annotations['location'] = np.array(locations)
    annotations['heading_angles'] = np.array(heading_angles)

    annotations['obj_ids'] = np.array(obj_ids)
    annotations['tracking_difficulty'] = np.array(tracking_difficulty)
    annotations['num_points_in_gt'] = np.array(num_points_in_gt)
    annotations['speed_x'] = np.array(speed_x)
    annotations['speed_y'] = np.array(speed_y)
    annotations['accel_x'] = np.array(acceleration_x)
    annotations['accel_y'] = np.array(acceleration_y)


    annotations = common_utils.drop_info_with_name(annotations, name='unknown')
    if annotations['name'].__len__() > 0:
        gt_boxes_lidar = np.concatenate([
            annotations['location'], annotations['dimensions'], annotations['heading_angles'][..., np.newaxis]],
            axis=1
        )
    else:
        gt_boxes_lidar = np.zeros((0, 7))
    annotations['gt_boxes_lidar'] = gt_boxes_lidar
    return annotations


def convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=0):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    points_NLZ = []
    points_intensity = []
    points_elongation = []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0])
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_mask = range_image_tensor[..., 0] > 0
        range_image_NLZ = range_image_tensor[..., 3]
        range_image_intensity = range_image_tensor[..., 1]
        range_image_elongation = range_image_tensor[..., 2]
        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local)

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
        points_tensor = tf.gather_nd(range_image_cartesian,
                                     tf.where(range_image_mask))
        points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
        points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
        points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))
        cp = camera_projections[c.name][0]
        cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))
        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())
        points_NLZ.append(points_NLZ_tensor.numpy())
        points_intensity.append(points_intensity_tensor.numpy())
        points_elongation.append(points_elongation_tensor.numpy())

    return points, cp_points, points_NLZ, points_intensity, points_elongation

def read_lidar(frame, r_i = 0 ):
    range_images, camera_projections, seg_labels,  range_image_top_pose = \
        frame_utils.parse_range_image_and_camera_projection(frame)

    points, cp_points, points_in_NLZ_flag, points_intensity, points_elongation = \
        convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=r_i)

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_in_NLZ_flag = np.concatenate(points_in_NLZ_flag, axis=0).reshape(-1, 1)
    points_intensity = np.concatenate(points_intensity, axis=0).reshape(-1, 1)
    points_elongation = np.concatenate(points_elongation, axis=0).reshape(-1, 1)

    num_points_of_each_lidar = [point.shape[0] for point in points]
    save_points = np.concatenate([
        points_all, points_intensity, points_elongation, points_in_NLZ_flag
    ], axis=-1).astype(np.float32)

    return save_points, num_points_of_each_lidar

def save_lidar_points(frame, cur_save_path):
    first_return, num_0 = read_lidar(frame, r_i=0)
    second_return, num_1 = read_lidar(frame, r_i=1)
    save_points = np.concatenate([first_return, second_return], 0)
    save_points = save_points.astype(np.float16)
    np.save(cur_save_path, save_points)
    # print('saving to ', cur_save_path)
    return num_0, num_1


def process_single_sequence(sequence_file, save_path, sampled_interval, has_label=True):

    sequence_name = os.path.splitext(os.path.basename(sequence_file))[0]
    cur_save_dir = save_path / sequence_name
    cur_save_dir.mkdir(parents=True, exist_ok=True)
    pkl_file = cur_save_dir / ('%s.pkl' % sequence_name)
    sequence_infos = []
    if pkl_file.exists():
        sequence_infos = pickle.load(open(pkl_file, 'rb'))
        print('Skip sequence since it has been processed before: %s' % pkl_file)
        return sequence_infos

    if not sequence_file.exists():
        print('NotFoundError: %s' % sequence_file)
        return []

    dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')

    cur_save_im_dir = cur_save_dir / 'image'
    cur_save_im_dir.mkdir(parents=True, exist_ok=True)


    for cnt, data in enumerate(dataset):
        if cnt % sampled_interval != 0:
            continue
        # print(sequence_name, cnt)
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        info = {}
        pc_info = {'num_features': 5, 'lidar_sequence': sequence_name, 'sample_idx': cnt}
        info['point_cloud'] = pc_info

        info['frame_id'] = sequence_name + ('_%03d' % cnt)
        image_info = {'image_shape':{},
                      'intrinsic_waymo':{},
                      'intrinsic':{},
                      'extrinsic': {},
                      'rolling_shutter_direction':{},
                      'camera_image_metadata':{}}

        cur_save_im_frame_dir = cur_save_dir / 'image'/str(cnt).zfill(4)
        cur_save_im_frame_dir.mkdir(parents=True, exist_ok=True)


        for j in range(5):
            cam_name = frame.context.camera_calibrations[j].name
            width = frame.context.camera_calibrations[j].width
            height = frame.context.camera_calibrations[j].height
            rolling_shutter_direction = frame.context.camera_calibrations[j].rolling_shutter_direction

            image_info['image_shape'][cam_name] = (height, width)

            intrinsic_waymo = np.array(frame.context.camera_calibrations[j].intrinsic, dtype=np.float32)
            intrinsic = np.zeros(shape=(3, 4))
            intrinsic[0, 0] = intrinsic_waymo[0]
            intrinsic[1, 1] = intrinsic_waymo[1]
            intrinsic[0, 2] = intrinsic_waymo[2]
            intrinsic[1, 2] = intrinsic_waymo[3]
            intrinsic[2, 2] = 1

            extrinsic = np.array(frame.context.camera_calibrations[j].extrinsic.transform, dtype=np.float32).reshape(4, 4)

            image_info['intrinsic_waymo'][cam_name] = intrinsic_waymo
            image_info['intrinsic'][cam_name] = intrinsic
            image_info['extrinsic'][cam_name] = extrinsic
            image_info['rolling_shutter_direction'][cam_name] = rolling_shutter_direction


        for index, image in enumerate(frame.images):
            cam_name = image.name
            camera_image_metadata = list(image.pose.transform)
            camera_image_metadata.append(image.velocity.v_x)
            camera_image_metadata.append(image.velocity.v_y)
            camera_image_metadata.append(image.velocity.v_z)
            camera_image_metadata.append(image.velocity.w_x)
            camera_image_metadata.append(image.velocity.w_y)
            camera_image_metadata.append(image.velocity.w_z)
            camera_image_metadata.append(image.pose_timestamp)
            camera_image_metadata.append(image.shutter)
            camera_image_metadata.append(image.camera_trigger_time)
            camera_image_metadata.append(image.camera_readout_done_time)
            image_info['camera_image_metadata'][cam_name] = camera_image_metadata

            cur_save_im_dir_sub = cur_save_im_frame_dir / (str(cam_name)+'.jpg')

            image_jpg = np.array(tf.image.decode_jpeg(image.image))

            temp = copy.deepcopy(image_jpg[:, :, 2])
            image_jpg[:, :, 2] = image_jpg[:, :, 0]
            image_jpg[:, :, 0] = temp

            cv2.imwrite(str(cur_save_im_dir_sub), image_jpg)


        info['image'] = image_info

        pose = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)
        info['pose'] = pose

        if has_label:
            annotations = generate_labels(frame)
            info['annos'] = annotations

        num_points_of_each_lidar_0,num_points_of_each_lidar_1 = save_lidar_points(frame, cur_save_dir / ('%04d.npy' % cnt))
        info['num_points_of_each_lidar_0'] = num_points_of_each_lidar_0
        info['num_points_of_each_lidar_1'] = num_points_of_each_lidar_1

        info['context_name'] = frame.context.name
        info['timestamp_micros'] = frame.timestamp_micros

        no_label_zone_list = []

        for waymo_no_label_zone in frame.no_label_zones:
            x = list(waymo_no_label_zone.x)
            y = list(waymo_no_label_zone.y)
            id = waymo_no_label_zone.id
            our_no_label_zone = {id:[x, y,]}
            no_label_zone_list.append(our_no_label_zone)

        info['no_label_zone_list'] = no_label_zone_list

        sequence_infos.append(info)

    with open(pkl_file, 'wb') as f:
        pickle.dump(sequence_infos, f)

    print('Infos are saved to (sampled_interval=%d): %s' % (sampled_interval, pkl_file))
    return sequence_infos

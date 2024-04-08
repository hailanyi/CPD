import tensorflow as tf

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils as fu
from waymo_open_dataset.camera.ops.py_camera_model_ops import world_to_image


C_FRONT = 1


if __name__ == '__main__':
    ds = tf.data.TFRecordDataset(
        '/home/asc01/data/waymo/perception/training/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord',
        compression_type=''
    )
    frame = dataset_pb2.Frame()
    frame.ParseFromString(next(iter(ds)).numpy())

    ris, cps,_, pps = fu.parse_range_image_and_camera_projection(frame)
    P, cp = fu.convert_range_image_to_point_cloud(frame, ris, cps, pps, 0)

    P = P[0]
    cp = cp[0]
    P_front = P[cp[..., 0] == C_FRONT]
    cp_front = cp[cp[..., 0] == C_FRONT]

    pose_tf = tf.reshape(tf.constant(frame.pose.transform), [4, 4])
    P_world = tf.einsum('ij,nj->ni', pose_tf[:3, :3], P_front) + pose_tf[:3, 3]
    P_world = tf.constant(P_world, dtype=tf.float64)

    cc = next(c for c in frame.context.camera_calibrations
              if c.name == C_FRONT)
    extrinsic = tf.reshape(tf.constant(cc.extrinsic.transform, dtype=tf.float64), [4, 4])
    intrinsic = tf.constant(cc.intrinsic, dtype=tf.float64)
    metadata = tf.constant([cc.width, cc.height, cc.rolling_shutter_direction], dtype=tf.float64)

    img = next(im for im in frame.images if im.name == C_FRONT)
    camera_image_metadata = tf.constant([
        *img.pose.transform,
        img.velocity.v_x, img.velocity.v_y, img.velocity.v_z,
        img.velocity.w_x, img.velocity.w_y, img.velocity.w_z,
        img.pose_timestamp,
        img.shutter,
        img.camera_trigger_time,
        img.camera_readout_done_time
    ], dtype=tf.float64)

    test = P_world[1000]
    groundtruth = cp_front[1000]
    proj = world_to_image(
        extrinsic, intrinsic, metadata, camera_image_metadata, test[None]
    )
    print(f'Projection: {proj[0, :2].numpy()}')
    print(f'Groundtruth: {groundtruth[1:3]}')
    # Prints:
    # Projection: [1467.3871  555.06  ]
    # Groundtruth: [1469  554]

    print(extrinsic)
    print(intrinsic)
    print(metadata)
    print(camera_image_metadata)
    print(P_world)


    proj_all = world_to_image(
        extrinsic, intrinsic, metadata, camera_image_metadata, P_world
    )

    print(proj_all)

    proj_err = tf.reduce_mean(tf.linalg.norm(
        proj_all[..., :2] - cp_front[..., 1:3], axis=-1
    ))

    print(f'Average Projection Error (all points): {proj_err} px.')
    # Prints:
    # Average Projection Error (all points): 5.581679344177246 px.
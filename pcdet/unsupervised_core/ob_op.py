import numpy as np
import copy


def to_sphere_coords(points):
    r = np.linalg.norm(points[:, 0:3], ord=2, axis=-1)
    theta = np.arccos(points[:, 2] / r)
    fan = np.arctan(points[:, 1] / points[:, 0]+0.001)

    new_points = copy.deepcopy(points)
    new_points[:, 0] = r
    new_points[:, 1] = theta
    new_points[:, 2] = fan

    return new_points

def la_sampling(points, vert_res=0.006, hor_res=0.003):
    new_points = copy.deepcopy(points)

    sp_coords = to_sphere_coords(new_points)

    voxel_dict = {}

    for i, point in enumerate(sp_coords):

        vert_coord = point[1] // vert_res
        hor_coord = point[2] // hor_res

        voxel_key = str(vert_coord) + '_' + str(hor_coord)

        if voxel_key in voxel_dict:

            voxel_dict[voxel_key]['sp'].append(point)
            voxel_dict[voxel_key]['pts'].append(new_points[i])
        else:
            voxel_dict[voxel_key] = {'sp': [point], 'pts': [new_points[i]]}

    sampled_list = []

    for voxel_key in voxel_dict:
        sp = voxel_dict[voxel_key]['sp']
        arg_min = np.argmin(np.array(sp)[:, 1])
        min_point = voxel_dict[voxel_key]['pts'][arg_min]
        sampled_list.append(min_point)
    new_points = np.array(sampled_list)
    if len(new_points) < 5:
        return points
    else:
        return new_points

def box_cut(box, cloud_in, scale=1.0):
    """
    input:
        box: array, shape=(7,)  (x, y, z, l, w, h, yaw)
        cloud: array, shape(N,M), (x, y, z, intensity, ...)
        scale: float, factor to enlarge the box size
    output:
        pts_in: array, points in box
        pts_out: array, points outside box
    """

    cloud = np.zeros(shape=(cloud_in.shape[0], 4))
    cloud[:, 0:3] = cloud_in[:, 0:3]
    cloud[:, 3] = 1

    x, y, z, l, w, h, yaw = box[0], box[1], box[2], box[3], box[4], box[5], box[6]

    trans_mat = np.eye(4, dtype=np.float32)
    trans_mat[0, 0] = np.cos(yaw)
    trans_mat[0, 1] = -np.sin(yaw)
    trans_mat[0, 3] = x
    trans_mat[1, 0] = np.sin(yaw)
    trans_mat[1, 1] = np.cos(yaw)
    trans_mat[1, 3] = y
    trans_mat[2, 3] = z

    trans_mat_i = np.linalg.inv(trans_mat)
    cloud = np.matmul(cloud, trans_mat_i.T)

    mask_l = np.logical_and(cloud[:, 0] < l * scale / 2, cloud[:, 0] > -l * scale / 2)
    mask_w = np.logical_and(cloud[:, 1] < w * scale / 2, cloud[:, 1] > -w * scale / 2)
    mask_h = np.logical_and(cloud[:, 2] < h * scale / 2, cloud[:, 2] > -h * scale / 2)
    mask = np.logical_and(np.logical_and(mask_w, mask_l), mask_h)
    mask_not = np.logical_not(mask)
    pts_in = cloud_in[mask]
    pts_out = cloud_in[mask_not]

    return pts_in, pts_out

def random_drop_out(points, rand_noise=0.2, offset=2):

    rand = np.random.choice([0, 1, 2, 3])
    new_points = []
    for i, p in enumerate(points):
        if rand == 0 and p[1] + np.random.randn() * rand_noise < offset:
            new_points.append(points[i])
        if rand == 1 and p[1] + np.random.randn() * rand_noise >= -offset:
            new_points.append(points[i])
        if rand == 2 and p[2] + np.random.randn() * rand_noise < offset:
            new_points.append(points[i])
        if rand == 3 and p[2] + np.random.randn() * rand_noise >= -offset:
            new_points.append(points[i])

    new_points = np.array(new_points)
    if len(new_points) <= 10:
        return points

    return new_points

def remove_past(lidar_points,
                objs,
                ids,
                cls,
                discard_cls=[0],
                not_care_box_cls=[4],
                sample_valid_cls=[1, 2, 3],
                sample_from_max_dis=12):

    discard_box = []
    not_care_box = []
    sample_box = []
    sample_ids = []
    sample_cls = []
    other_box = []
    other_ids = []
    other_cls = []

    for i, ob in enumerate(objs):
        if cls[i] in discard_cls:
            discard_box.append(ob)
        elif cls[i] in not_care_box_cls:
            not_care_box.append(ob)
        elif cls[i] in sample_valid_cls and np.linalg.norm(ob[0:3]) < sample_from_max_dis:
            sample_box.append(ob)
            sample_ids.append(ids[i])
            sample_cls.append(cls[i])
        else:
            other_box.append(ob)
            other_ids.append(ids[i])
            other_cls.append(cls[i])

    sample_points_base = []

    # for box in not_care_box:
    #    _, lidar_points = box_cut(box, lidar_points, scale=1.3)

    for box in discard_box:
        _, lidar_points = box_cut(box, lidar_points)

    for box in sample_box:
        points_in, points_out = box_cut(box, lidar_points)
        this_points = copy.deepcopy(points_in)
        this_points[:, 0:3] -= box[0:3]
        sample_points_base.append(this_points)

    changed_points = []
    changed_box = []
    changed_ids = []
    changed_cls = []

    if len(sample_box)==0:
        objs = np.array(sample_box + other_box)
        ids = np.array(sample_ids + other_ids)
        cls = np.array(sample_cls + other_cls)
        return lidar_points, objs, ids, cls

    for i, dis_box in enumerate(discard_box):
        random_int = np.random.randint(0, len(sample_box))
        this_sample_box = copy.deepcopy(sample_box[random_int])
        this_sample_points = copy.deepcopy(sample_points_base[random_int])
        this_ids = copy.deepcopy(sample_ids[random_int])
        this_cls = copy.deepcopy(sample_cls[random_int])

        this_sample_box[0:3] = dis_box[0:3]
        this_sample_points[:, 0:3] += dis_box[0:3]

        this_sample_points = la_sampling(this_sample_points)
        this_sample_points = random_drop_out(this_sample_points)

        changed_points.append(this_sample_points)
        changed_box.append(this_sample_box)
        changed_ids.append(this_ids)
        changed_cls.append(this_cls)

    lidar_points = np.concatenate(changed_points + [lidar_points])
    objs = np.array(changed_box + sample_box + other_box)
    ids = np.array(changed_ids + sample_ids + other_ids)
    cls = np.array(changed_cls + sample_cls + other_cls)

    return lidar_points, objs, ids, cls

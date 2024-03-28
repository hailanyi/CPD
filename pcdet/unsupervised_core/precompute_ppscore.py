import os
import os.path as osp
import pickle as pkl
import numpy as np
from scipy.spatial import Delaunay, cKDTree
import copy

def count_neighbors(ptc, trees, max_neighbor_dist=0.3):
    neighbor_count = {}
    for seq in trees.keys():
        neighbor_count[seq] = trees[seq].query_ball_point(
            ptc[:, :3], r=max_neighbor_dist,
            return_length=True)
    return np.stack(list(neighbor_count.values())).T

def compute_ephe_score(count):
    N = count.shape[1]
    P = count / (np.expand_dims(count.sum(axis=1), -1) + 1e-8)
    H = (-P * np.log(P + 1e-8)).sum(axis=1) / np.log(N)

    return H

def compute_ppscore(cur_frame, neighbor_traversals=None, max_neighbor_dist = 0.3):

    trees = {}

    for seq_id, points in enumerate(neighbor_traversals):
        trees[seq_id] = cKDTree(points)

    count = count_neighbors(cur_frame, trees, max_neighbor_dist)

    H = compute_ephe_score(count)

    return H

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


def save_pp_score(seq_name, root_path, max_win=30, win_inte=5, max_neighbor_dist=0.3):


    file_path = os.path.join(root_path, seq_name, 'ppscore')


    pkl_path = os.path.join(root_path, seq_name, seq_name + '.pkl')

    all_lidar = os.listdir(os.path.join(root_path, seq_name))
    all_lidar.sort()

    with open(pkl_path, 'rb') as f:
        infos = pkl.load(f)


    win_size = 1

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    for i in range(0, len(infos)):
        pose_i = np.linalg.inv(infos[i]['pose'])

        all_traversals = []

        cur_points = None

        max_tra = max_win

        for j in range(i-max_tra, i+max_tra, win_inte):

            this_tra = []
            for k in range(j, j+win_size):
                info_path = str(j).zfill(4) + '.npy'
                lidar_path = os.path.join(root_path, seq_name, info_path)
                if not os.path.exists(lidar_path):
                    continue
                lidar_points = np.load(lidar_path)[:, 0:3]

                if k == i:
                    cur_points = lidar_points

                pose_k = infos[k]['pose']
                this_tra.append(points_rigid_transform(lidar_points, pose_k))
            if len(this_tra)>0:
                this_tra = np.concatenate(this_tra)
                all_traversals.append(points_rigid_transform(this_tra, pose_i))

        H = compute_ppscore(cur_points, all_traversals, max_neighbor_dist=max_neighbor_dist)

        this_file = os.path.join(file_path, str(i).zfill(4) + '.npy')

        np.save(this_file, np.array(H).astype(np.float16))

    return True
from pcdet.unsupervised_core.ground_removal import Processor
import numpy as np
import os
import pickle as pkl
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from pcdet.unsupervised_core.tracker.tracker import Tracker3D
from pcdet.unsupervised_core.tracker.box_op import register_bbs
from scipy.spatial import cKDTree
import copy
import warnings
warnings.filterwarnings("ignore")


def sigmoid(x):
    return 1.0 / (1 + np.exp(-float(x)))

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum(axis=0)

def norm(z):
    return z/z.sum()

def KL_entropy_score(x, y, max_dif = 0.05):
    KL = 0.0
    for i in range(len(x)):
        KL += x[i] * np.log(x[i] / y[i])

    if KL>max_dif:
        KL = max_dif
    return (max_dif-KL)/max_dif

def angle_from_vector(x,y):

    if x>0:
        return np.arctan(y/x)
    else:
        return np.pi+np.arctan(y/x)

def density_guided_drift(points, ori_box):

    new_box = copy.deepcopy(ori_box)

    x, y, z, l, w, h, yaw = ori_box[0], ori_box[1], ori_box[2], ori_box[3], ori_box[4], ori_box[5], ori_box[6]

    cloud = np.zeros(shape=(points.shape[0], 4))
    cloud[:, 0:3] = points[:, 0:3]
    cloud[:, 3] = 1

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

    max_x = np.max(cloud[:, 0])
    min_x = np.min(cloud[:, 0])
    max_y = np.max(cloud[:, 1])
    min_y = np.min(cloud[:, 1])

    mask_x = cloud[:,0]>0#(max_x-min_x)/2
    mask_y = cloud[:,1]>0#(max_y-min_y)/2

    center_point = np.array([[0.,0.,0.,1.]])

    if mask_x.sum()/mask_x.shape[0]>1/2:
        new_x = l/2-max_x
        center_point[0, 0] = -new_x
    else:
        new_x = -l/2-min_x
        center_point[0, 0] = -new_x

    if mask_y.sum()/mask_y.shape[0]>1/2:
        new_y = w/2-max_y
        center_point[0, 1] = -new_y
    else:
        new_y = -w/2-min_y
        center_point[0, 1] = -new_y

    center_point = np.matmul(center_point, trans_mat.T)

    new_box[0] = center_point[0,0]
    new_box[1] = center_point[0,1]

    return np.array(new_box)

def corner_align(box, l_off, w_off):
    x, y, z, l, w, h, yaw = box[0], box[1], box[2], box[3], box[4], box[5], box[6]

    trans_mat = np.eye(4, dtype=np.float32)
    trans_mat[0, 0] = np.cos(yaw)
    trans_mat[0, 1] = -np.sin(yaw)
    trans_mat[0, 3] = x
    trans_mat[1, 0] = np.sin(yaw)
    trans_mat[1, 1] = np.cos(yaw)
    trans_mat[1, 3] = y
    trans_mat[2, 3] = z

    corner1 = np.array([[l_off / 2, w_off / 2, 0, 1]])
    corner2 = np.array([[-l_off / 2, -w_off / 2, 0, 1]])
    corner3 = np.array([[l_off / 2, -w_off / 2, 0, 1]])
    corner4 = np.array([[-l_off / 2, w_off / 2, 0, 1]])

    all_corners = np.concatenate([corner1, corner2, corner3, corner4], 0)

    new_corners = np.matmul(all_corners, trans_mat.T)

    dis = np.linalg.norm(new_corners, axis=-1)

    arg_min = np.argmax(dis)

    box[3] += l_off
    box[4] += w_off
    box[0:3] = new_corners[arg_min, 0:3]

    return box



def correct_orientation(points, box_in):
    box = copy.deepcopy(box_in)

    x, y, z, l, w, h, yaw = box[0], box[1], box[2], box[3], box[4], box[5], box[6]

    cloud = np.zeros(shape=(points.shape[0], 4))
    cloud[:, 0:3] = points[:, 0:3]
    cloud[:, 3] = 1

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


    min_x = np.min(cloud[:,0])
    max_x = np.max(cloud[:,0])

    min_y = np.min(cloud[:,1])
    max_y = np.max(cloud[:,1])
    parts = 7
    # correct based on x
    if ((max_x-min_x)/l)*2>((max_y-min_y)/w):

        mid_x = (max_x-min_x)/2.+min_x
        top_pts = cloud[cloud[:, 0] > mid_x]
        bot_pts = cloud[cloud[:, 0] < mid_x]
        y_mask = cloud[:,1]>0


        delta = (max_x-mid_x)/parts

        if y_mask.sum()/y_mask.shape[0]>1/2:

            all_max_top = []

            for i in range(parts):
                this_top_i = mid_x+i*delta
                this_top_i_1 = mid_x+(i+1)*delta
                mask_i = top_pts[:,0]>this_top_i
                mask_i_1 = top_pts[:,0]<=this_top_i_1
                this_mask = mask_i*mask_i_1
                valid_pts = top_pts[this_mask]
                if len(valid_pts)!=0:
                    arg_max_y_top = np.argmax(valid_pts[:,1])
                    max_top_pt = valid_pts[arg_max_y_top]
                    all_max_top.append(max_top_pt)

            all_max_bot = []
            for i in range(parts):
                this_bot_i = min_x+i*delta
                this_bot_i_1 = min_x+(i+1)*delta
                mask_i = bot_pts[:,0]>this_bot_i
                mask_i_1 = bot_pts[:,0]<=this_bot_i_1
                this_mask = mask_i*mask_i_1
                valid_pts = bot_pts[this_mask]
                if len(valid_pts)!=0:
                    arg_max_y_bot = np.argmax(valid_pts[:, 1])
                    max_bot_pt = valid_pts[arg_max_y_bot]
                    all_max_bot.append(max_bot_pt)
            if len(all_max_top)>0 and len(all_max_bot)>0:
                all_max_top = np.mean(np.array(all_max_top),0)
                all_max_bot = np.mean(np.array(all_max_bot),0)

                delta = np.arctan((all_max_top[1]-all_max_bot[1])/(all_max_top[0]-all_max_bot[0]))
                box[6]+=delta
            return box

        else:

            all_max_top = []

            for i in range(parts):
                this_top_i = mid_x + i * delta
                this_top_i_1 = mid_x + (i + 1) * delta
                mask_i = top_pts[:, 0] > this_top_i
                mask_i_1 = top_pts[:, 0] <= this_top_i_1
                this_mask = mask_i * mask_i_1
                valid_pts = top_pts[this_mask]
                if len(valid_pts)!=0:
                    arg_max_y_top = np.argmin(valid_pts[:, 1])
                    max_top_pt = valid_pts[arg_max_y_top]
                    all_max_top.append(max_top_pt)

            all_max_bot = []
            for i in range(parts):
                this_bot_i = min_x + i * delta
                this_bot_i_1 = min_x + (i + 1) * delta
                mask_i = bot_pts[:, 0] > this_bot_i
                mask_i_1 = bot_pts[:, 0] <= this_bot_i_1
                this_mask = mask_i * mask_i_1
                valid_pts = bot_pts[this_mask]
                if len(valid_pts)!=0:
                    arg_max_y_bot = np.argmin(valid_pts[:, 1])
                    max_bot_pt = valid_pts[arg_max_y_bot]
                    all_max_bot.append(max_bot_pt)

            if len(all_max_top)>0 and len(all_max_bot)>0:
                all_max_top = np.mean(np.array(all_max_top), 0)
                all_max_bot = np.mean(np.array(all_max_bot), 0)

                delta = np.arctan((all_max_top[1] - all_max_bot[1]) / (all_max_top[0] - all_max_bot[0]))
                box[6]+=delta

            return box
    else:
        mid_y = (max_y - min_y) / 2. + min_y
        top_pts = cloud[cloud[:, 1] > mid_y]
        bot_pts = cloud[cloud[:, 1] < mid_y]
        x_mask = cloud[:, 0] > 0

        delta = (max_y - mid_y) / parts

        if x_mask.sum() / x_mask.shape[0] > 1 / 2:

            all_max_top = []

            for i in range(parts):
                this_top_i = mid_y + i * delta
                this_top_i_1 = mid_y + (i + 1) * delta
                mask_i = top_pts[:, 1] > this_top_i
                mask_i_1 = top_pts[:, 1] <= this_top_i_1
                this_mask = mask_i * mask_i_1
                valid_pts = top_pts[this_mask]
                if len(valid_pts) != 0:
                    arg_max_y_top = np.argmax(valid_pts[:, 0])
                    max_top_pt = valid_pts[arg_max_y_top]
                    all_max_top.append(max_top_pt)

            all_max_bot = []
            for i in range(parts):
                this_bot_i = min_y + i * delta
                this_bot_i_1 = min_y + (i + 1) * delta
                mask_i = bot_pts[:, 1] > this_bot_i
                mask_i_1 = bot_pts[:, 1] <= this_bot_i_1
                this_mask = mask_i * mask_i_1
                valid_pts = bot_pts[this_mask]
                if len(valid_pts) != 0:
                    arg_max_y_bot = np.argmax(valid_pts[:, 0])
                    max_bot_pt = valid_pts[arg_max_y_bot]
                    all_max_bot.append(max_bot_pt)

            if len(all_max_top) == 0 or len(all_max_bot)==0:
                return box
            else:

                all_max_top = np.mean(np.array(all_max_top), 0)
                all_max_bot = np.mean(np.array(all_max_bot), 0)


                delta = np.arctan((all_max_top[0] - all_max_bot[0]) / (all_max_top[1] - all_max_bot[1]))
                box[6] += delta
                return box

        else:

            all_max_top = []

            for i in range(parts):
                this_top_i = mid_y + i * delta
                this_top_i_1 = mid_y + (i + 1) * delta
                mask_i = top_pts[:, 1] > this_top_i
                mask_i_1 = top_pts[:, 1] <= this_top_i_1
                this_mask = mask_i * mask_i_1
                valid_pts = top_pts[this_mask]
                if len(valid_pts) != 0:
                    arg_max_y_top = np.argmin(valid_pts[:, 0])
                    max_top_pt = valid_pts[arg_max_y_top]
                    all_max_top.append(max_top_pt)

            all_max_bot = []
            for i in range(parts):
                this_bot_i = min_y + i * delta
                this_bot_i_1 = min_y + (i + 1) * delta
                mask_i = bot_pts[:, 1] > this_bot_i
                mask_i_1 = bot_pts[:, 1] <= this_bot_i_1
                this_mask = mask_i * mask_i_1
                valid_pts = bot_pts[this_mask]
                if len(valid_pts) != 0:
                    arg_max_y_bot = np.argmin(valid_pts[:, 0])
                    max_bot_pt = valid_pts[arg_max_y_bot]
                    all_max_bot.append(max_bot_pt)

            if len(all_max_top) == 0 or len(all_max_bot)==0:
                return box
            else:
                all_max_top = np.mean(np.array(all_max_top), 0)
                all_max_bot = np.mean(np.array(all_max_bot), 0)

                delta = np.arctan((all_max_top[0] - all_max_bot[0]) / (all_max_top[1] - all_max_bot[1]))
                box[6] += delta

                return box

def points_rigid_transform(cloud, pose):
    cloud = np.array(cloud)
    if cloud.shape[0] == 0:
        return cloud
    mat = np.ones(shape=(cloud.shape[0], 4), dtype=np.float32)
    pose_mat = np.mat(pose)
    mat[:, 0:3] = cloud[:, 0:3]
    mat = np.mat(mat)
    transformed_mat = pose_mat * mat.T
    T = np.array(transformed_mat.T, dtype=np.float32)
    return T[:, 0:3]

def get_registration_angle(mat):

    cos_theta=mat[0,0]
    sin_theta=mat[1,0]

    if  cos_theta < -1:
        cos_theta = -1
    if cos_theta > 1:
        cos_theta = 1

    theta_cos = np.arccos(cos_theta)

    if sin_theta >= 0:
        return theta_cos
    else:
        return 2 * np.pi - theta_cos

def box_rigid_transform(in_box, pose_pre, pose_cur):

    inv_pose_of_last_frame = np.linalg.inv(pose_cur)
    registration_mat = np.matmul(inv_pose_of_last_frame, pose_pre)
    box = copy.deepcopy(in_box)
    angle = get_registration_angle(registration_mat)
    box[0:3] = points_rigid_transform(np.array([box[0:3]]),registration_mat)[0,0:3]
    box[6]+=angle

    return box

def voxel_sampling(point2, res_x=0.1, res_y=0.1, res_z = 0.1):

    min_x = point2[:,0].min()
    min_y = point2[:,1].min()
    min_z = point2[:,2].min()

    voxels = {}

    for point in point2:
        x = point[0]
        y = point[1]
        z = point[2]

        x_coord = (x-min_x)//res_x
        y_coord = (y-min_y)//res_y
        z_coord = (z-min_z)//res_z

        key = str(x_coord)+'_'+str(y_coord)+'_'+str(z_coord)

        voxels[key] = point

    return np.array(list(voxels.values()))

def smooth_points(points, rad = 0.2):
    trees = cKDTree(points[:, 0:3])
    num = trees.query_ball_point(
        points[:, 0:3], r=rad,
        return_length=True)
    return points[num>3]

def compute_confidence(points, box, parts=6):
    x, y, z, l, w, h, yaw = box[0], box[1], box[2], box[3], box[4], box[5], box[6]

    cloud = np.zeros(shape=(points.shape[0], 4))
    cloud[:, 0:3] = points[:, 0:3]
    cloud[:, 3] = 1

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

    delta_l = l/parts
    delta_w = w/parts

    valid_vol = 0

    for i in range(parts):
        for j in range(parts):
            mask_x_l = -l/2+i*delta_l<=cloud[:, 0]
            mask_x_r = cloud[:,0]<-l/2+(i+1)*delta_l
            mask_y_l = -w/2+j*delta_w<=cloud[:, 1]
            mask_y_r = cloud[:, 1]<-w/2+(j+1)*delta_w

            mask = mask_x_l*mask_x_r*mask_y_l*mask_y_r

            this_pts = cloud[mask]

            if len(this_pts)>1:
                valid_vol+=1

    return valid_vol/(parts**2)

def hierarchical_occupancy_score(points, box, parts=[7,5,3]):
    all_confi = 0
    for part in parts:
        all_confi+=compute_confidence(points,box,part)
    return all_confi/len(parts)

def correct_heading(orin_points, box, parts=10):
    x, y, z, l, w, h, yaw = box[0, 0], box[0, 1], box[0, 2], box[0, 3], box[0, 4], box[0, 5], box[0, 6]
    cloud = np.zeros(shape=(orin_points.shape[0], 4))
    cloud[:, 0:3] = orin_points[:, 0:3]
    cloud[:, 3] = 1
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

    delta_l = l / parts

    z_x_max = []
    z_x_min = []

    for i in range(parts):
        mask_x_l = -l / 2 + i * delta_l <= cloud[:, 0]
        mask_x_r = cloud[:, 0] < -l / 2 + (i + 1) * delta_l
        mask = mask_x_l*mask_x_r
        this_pts_cloud = cloud[mask]
        if -l / 2 + i * delta_l<0 and len(this_pts_cloud)>0:
            z_x_min.append(np.max(this_pts_cloud[:,2]))
        if -l / 2 + (i + 1) * delta_l>0 and len(this_pts_cloud)>0:
            z_x_max.append(np.max(this_pts_cloud[:, 2]))

    if len(z_x_max)==0:
        z_x_max.append(0)
    if len(z_x_min)==0:
        z_x_min.append(0)

    if np.mean(z_x_min)<np.mean(z_x_max):
        new_box = copy.deepcopy(box)
        new_box[0, 6]+=np.pi
        return new_box
    else:
        return box

def drop_cls(boxes, name, ids=None, dif=None, confi=None, proto_id=None, droped_cls=['Dis_Small', 'Dis_Large']):

    for cls_name in droped_cls:
        mask = name!=cls_name

        boxes = boxes[mask]
        if ids is not None:
            ids = ids[mask]
        if dif is not None:
            dif = dif[mask]
        if confi is not None:
            confi=confi[mask]
        if proto_id is not None:
            proto_id = proto_id[mask]
        name = name[mask]

    return boxes, name, ids, dif, confi, proto_id


class OutlineFitter:
    def __init__(self, sensor_height=0,
                 ground_min_threshold=[0.2, -0.2, -0.5],
                 ground_min_distance=[0, 20, 40, 100],
                 ground_max_threshold=1,
                 cluster_dis = 0.5,
                 cluster_min_points=40,
                 discard_max_height=4,
                 min_box_volume = 0.3,
                 min_box_height = 0.5,
                 max_box_volume = 200,
                 max_box_len = 10
                 ):

        self.sensor_height = sensor_height
        self.ground_min_threshold = ground_min_threshold  # discard all the low points
        self.ground_min_distance = ground_min_distance  # discard all the low points for distance
        self.ground_max_threshold = ground_max_threshold  # keep all the top points
        self.clutter_dis = cluster_dis
        self.clutter_min_points = cluster_min_points
        self.discard_max_height = discard_max_height
        self.min_box_volume = min_box_volume
        self.min_box_hight = min_box_height
        self.max_box_volume = max_box_volume
        self.max_box_len = max_box_len

        self.cluster_method = DBSCAN(self.clutter_dis, min_samples=10)

    def compute_volume(self, boxes):
        w = boxes[:, 3]
        l = boxes[:, 4]
        h = boxes[:, 5]
        volume = np.multiply(np.multiply(w, l), h)

        return volume

    def remove_ground(self, points):

        hight_points = points[points[:, 2]>=self.ground_max_threshold]

        low_points = points[points[:, 2] < self.ground_max_threshold]

        process = Processor(n_segments=150, n_bins=150, line_search_angle=0.3, max_dist_to_line=0.1,
                            sensor_height=self.sensor_height, max_start_height=0.5, long_threshold=8)
        vel_non_ground = process(low_points)

        non_ground_points = np.concatenate([hight_points[:, 0:3], vel_non_ground[:,0:3]], 0)

        distance = np.linalg.norm(non_ground_points[:,0:3], axis=1)

        all_points = []

        for i in range(len(self.ground_min_threshold)):
            if i==0:
                mask_dis = distance<self.ground_min_distance[1]
            elif i==len(self.ground_min_threshold)-1:
                mask_dis = distance > self.ground_min_distance[i]
            else:
                mask_max = distance < self.ground_min_distance[i+1]
                mask_min = distance > self.ground_min_distance[i]
                mask_dis = mask_max*mask_min

            this_points = non_ground_points[mask_dis]
            this_points = this_points[this_points[:,2]>self.ground_min_threshold[i]]

            all_points.append(this_points)

        non_ground_points = np.concatenate(all_points)


        return non_ground_points

    def distance_point_to_lines(self, points, lines_begin, lines_end):

        # Get the number of points and lines
        n = points.shape[0]
        m = lines_begin.shape[0]
        # Compute the direction vectors of the lines
        pq = lines_end - lines_begin
        # Compute the normal vectors of the lines
        n = np.array([-pq[:, 1], pq[:, 0]]).transpose()
        # Compute the distance from each point to each line
        P = points[:, np.newaxis, :]
        Q = lines_begin[:, :]
        PQ = pq[np.newaxis, :, :]
        N = n[np.newaxis, :, :]

        distance = np.abs(np.cross(P - Q, pq[np.newaxis, :, :], axis=2)) / np.linalg.norm(pq, axis=1)
        # Take the minimum distance for each point
        return distance

    def distances_to_rectangles(self, points_matrix, vertices_matrix):

        distances = self.distance_point_to_lines(points_matrix, vertices_matrix[:, 0, :], vertices_matrix[:, 1, :])
        distances = np.minimum(distances, self.distance_point_to_lines(points_matrix, vertices_matrix[:, 1, :],
                                                                  vertices_matrix[:, 2, :]))
        distances = np.minimum(distances, self.distance_point_to_lines(points_matrix, vertices_matrix[:, 2, :],
                                                                  vertices_matrix[:, 3, :]))
        distances = np.minimum(distances, self.distance_point_to_lines(points_matrix, vertices_matrix[:, 3, :],
                                                                  vertices_matrix[:, 0, :]))

        return distances

    def minimum_bounding_rectangle_distance(self,points):
        """
        Find the smallest bounding rectangle for a set of points.
        Returns a set of points representing the corners of the bounding box.

        :param points: an nx2 matrix of coordinates
        :rval: an nx2 matrix of coordinates
        """

        pi2 = np.pi / 2.

        # get the convex hull for the points
        # print('#######',points)
        hull_points = points[ConvexHull(points).vertices]

        # calculate edge angles
        edges = np.zeros((len(hull_points) - 1, 2))
        edges = hull_points[1:] - hull_points[:-1]

        angles = np.zeros((len(edges)))
        angles = np.arctan2(edges[:, 1], edges[:, 0])

        angles = np.abs(np.mod(angles, pi2))
        angles = np.unique(angles)

        # find rotation matrices
        # XXX both work
        rotations = np.vstack([
            np.cos(angles),
            np.cos(angles - pi2),
            np.cos(angles + pi2),
            np.cos(angles)]).T
        #     rotations = np.vstack([
        #         np.cos(angles),
        #         -np.sin(angles),
        #         np.sin(angles),
        #         np.cos(angles)]).T
        rotations = rotations.reshape((-1, 2, 2))

        # apply rotations to the hull
        rot_points = np.dot(rotations, hull_points.T)

        # find the bounding points
        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)

        min_xy = np.concatenate((np.array(min_x).reshape(-1, 1), np.array(min_y).reshape(-1, 1)), -1)  # n*2
        minx_max_y = np.concatenate((np.array(min_x).reshape(-1, 1), np.array(max_y).reshape(-1, 1)), -1)
        maxx_min_y = np.concatenate((np.array(max_x).reshape(-1, 1), np.array(min_y).reshape(-1, 1)), -1)
        max_xy = np.concatenate((np.array(max_x).reshape(-1, 1), np.array(max_y).reshape(-1, 1)), -1)

        # find the box with the best area
        areas = (max_x - min_x) * (max_y - min_y) * 0.5

        value_list = []
        for i in range(len(min_x)):
            # pts_ = rot_points[i, :, :]

            rect = np.concatenate((min_xy[i, :], minx_max_y[i, :]), 0)
            rect = np.concatenate((rect, max_xy[i, :]), 0)
            rect = np.concatenate((rect, maxx_min_y[i, :]), 0)
            rect = rect.reshape(-1, 4, 2)
            points_to_edges = self.distances_to_rectangles(rot_points[i, :, :].T, rect)
            value = np.mean(points_to_edges) * 0.5
            # value = np.mean(point_to_rectangles_distance_corner(rot_points[i, :, :].T, rect), 0)
            value_list.append(value)

        value_list = np.array(value_list)

        areas = (areas-areas.min())/(areas.max()-areas.min()+0.0001)
        value_list = (value_list-value_list.min())/(value_list.max()-value_list.min()+0.0001)

        value_list = value_list+areas

        best_idx = np.argmin(value_list)
        # best_idx = np.argmin(value_list + areas)

        # return the best box
        x1 = max_x[best_idx]
        x2 = min_x[best_idx]
        y1 = max_y[best_idx]
        y2 = min_y[best_idx]
        r = rotations[best_idx]

        rval = np.zeros((4, 2))
        rval[0] = np.dot([x1, y2], r)
        rval[1] = np.dot([x2, y2], r)
        rval[2] = np.dot([x2, y1], r)
        rval[3] = np.dot([x1, y1], r)

        return rval, angles[best_idx], areas[best_idx]

    def minimum_bounding_rectangle(self, points):

        pi2 = np.pi / 2.

        # get the convex hull for the points
        # print('#######',points)
        hull_points = points[ConvexHull(points).vertices]

        # calculate edge angles
        edges = np.zeros((len(hull_points) - 1, 2))
        edges = hull_points[1:] - hull_points[:-1]

        angles = np.zeros((len(edges)))
        angles = np.arctan2(edges[:, 1], edges[:, 0])

        angles = np.abs(np.mod(angles, pi2))
        angles = np.unique(angles)

        # find rotation matrices
        # XXX both work
        rotations = np.vstack([
            np.cos(angles),
            np.cos(angles - pi2),
            np.cos(angles + pi2),
            np.cos(angles)]).T

        rotations = rotations.reshape((-1, 2, 2))

        # apply rotations to the hull
        rot_points = np.dot(rotations, hull_points.T)

        # find the bounding points
        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)

        # find the box with the best area
        areas = (max_x - min_x) * (max_y - min_y)


        best_idx = np.argmin(areas)

        # return the best box
        x1 = max_x[best_idx]
        x2 = min_x[best_idx]
        y1 = max_y[best_idx]
        y2 = min_y[best_idx]
        r = rotations[best_idx]

        rval = np.zeros((4, 2))
        rval[0] = np.dot([x1, y2], r)
        rval[1] = np.dot([x2, y2], r)
        rval[2] = np.dot([x2, y1], r)
        rval[3] = np.dot([x1, y1], r)

        return rval, angles[best_idx], areas[best_idx]

    def get_obj(self, ptc):

        points = ptc[:, [0, 1]]
        new_points = np.array([points[:, 1], points[:, 0]]).T
        corners, ry, area = self.minimum_bounding_rectangle_distance(new_points)
        ry *= -1
        l = np.linalg.norm(corners[0] - corners[1])
        w = np.linalg.norm(corners[0] - corners[-1])
        c = (corners[0] + corners[2]) / 2
        bottom = ptc[:, 2].max()
        # bottom = full_ptc[:, 2].max()

        # bottom = get_lowest_point_rect(full_ptc, c, l, w, ry)
        h = bottom - ptc[:, 2].min()
        # obj = types.SimpleNamespace()
        obj = np.empty((1, 7))

        obj[:, 0] = c[1]
        obj[:, 1] = c[0]
        obj[:, 2] = bottom - h / 2
        obj[:, 3] = w
        obj[:, 4] = l
        obj[:, 5] = h
        obj[:, 6] = ry

        return obj


    def clustering(self, points):

        self.cluster_method.fit(points)

        num_instance = len(set(self.cluster_method.labels_)) - (1 if -1 in self.cluster_method.labels_ else 0)

        all_valid_clusters = []
        labels = []

        for i in range(num_instance):

            this_points = points[self.cluster_method.labels_==i]

            if len(this_points) > self.clutter_min_points and this_points[:, 2].max()<self.discard_max_height:

                all_valid_clusters.append(this_points)
                labels.append(self.cluster_method.labels_[self.cluster_method.labels_==i])

        return all_valid_clusters, labels

    def box_fit(self, points_list, offset=0.2):

        all_box = []
        for i, points in enumerate(points_list):

            min_height = points[:, 2].min()
            filter_low_mask = (points[:, 2]>(min_height+offset))

            points = points[filter_low_mask]

            try:
                box = self.get_obj(points)
            except:
                continue

            box[0, 2] -= (offset / 2)
            box[0, 5] += offset
            vl = self.compute_volume(box)

            l = max(box[0, 3],box[0, 4])

            if np.linalg.norm(box[0,0:3])<self.ground_min_distance[1]:
                box[0, 2]-=self.ground_min_threshold[0]/2
                box[0, 5]+=self.ground_min_threshold[0]

            if vl[0]>self.min_box_volume and box[0, 5] > self.min_box_hight and vl[0]<self.max_box_volume and l<self.max_box_len:

                if box[0, 3] < box[0, 4]:
                    temp = box[0, 3]
                    box[0, 3] = box[0, 4]
                    box[0, 4] = temp
                    box[0, 6] += np.pi / 2
                all_box.append(box)

        if len(all_box)==0:
            return all_box
        else:
            return np.concatenate(all_box)

    def box_fit_DGD(self, points_list, offset=0.2):

        all_box = []
        for i, points in enumerate(points_list):

            min_height = points[:, 2].min()
            filter_low_mask = (points[:, 2]>(min_height+offset))

            points = points[filter_low_mask]

            try:
                box = self.get_obj(points)
            except:
                continue

            box[0, 2] -= (offset / 2)
            box[0, 5] += offset
            vl = self.compute_volume(box)

            l = max(box[0, 3],box[0, 4])

            if np.linalg.norm(box[0,0:3])<self.ground_min_distance[1]:
                box[0, 2]-=self.ground_min_threshold[0]/2
                box[0, 5]+=self.ground_min_threshold[0]

            if vl[0]>self.min_box_volume and box[0, 5] > self.min_box_hight and vl[0]<self.max_box_volume and l<self.max_box_len:

                if box[0, 3] < box[0, 4]:
                    temp = box[0, 3]
                    box[0, 3] = box[0, 4]
                    box[0, 4] = temp
                    box[0, 6] += np.pi / 2

                box[0, :] = density_guided_drift(points, box[0, :])
                box[0, :] = correct_orientation(points, box[0, :])
                box = correct_heading(points, box)
                all_box.append(box)

        if len(all_box)==0:
            return all_box
        else:
            return np.concatenate(all_box)

    def get_box_cls(self, boxes, config, return_name=True):

        ob_cls = []
        ob_dif = []

        cls_proto = config.cls

        cls_proto_L = config.cls_L

        cls_proto_W = config.cls_W

        cls_proto_H = config.cls_H

        for box in boxes:

            ob_dif.append(1)

            l, w, h = box[3], box[4], box[ 5]
            top_z = box[2] + h / 2

            if top_z > config.max_top_z or w > config.max_width \
                    or l > config.max_len:
                if return_name:
                    ob_cls.append('Dis_Large')
                else:
                    ob_cls.append(cls_proto['Dis_Large'])
            elif cls_proto_L['Dis_Small'][0] < l <= cls_proto_L['Dis_Small'][1] \
                    and cls_proto_H['Dis_Small'][0] < h <= cls_proto_H['Dis_Small'][1] \
                    and cls_proto_W['Dis_Small'][0] < w <= cls_proto_W['Dis_Small'][1]:
                if return_name:
                    ob_cls.append('Dis_Small')
                else:
                    ob_cls.append(cls_proto['Dis_Small'])
            elif cls_proto_L['Pedestrian'][0] < l <= cls_proto_L['Pedestrian'][1] \
                    and cls_proto_H['Pedestrian'][0] < h <= cls_proto_H['Pedestrian'][1] \
                    and cls_proto_W['Pedestrian'][0] < w <= cls_proto_W['Pedestrian'][1]:
                if return_name:
                    ob_cls.append('Pedestrian')
                else:
                    ob_cls.append(cls_proto['Pedestrian'])
            elif cls_proto_L['Cyclist'][0] < l <= cls_proto_L['Cyclist'][1] \
                    and cls_proto_H['Cyclist'][0] < h <= cls_proto_H['Cyclist'][1] \
                    and cls_proto_W['Cyclist'][0] < w <= cls_proto_W['Cyclist'][1]:
                if return_name:
                    ob_cls.append('Cyclist')
                else:
                    ob_cls.append(cls_proto['Cyclist'])
            elif cls_proto_L['Vehicle'][0] < l <= cls_proto_L['Vehicle'][1] \
                    and cls_proto_H['Vehicle'][0] < h <= cls_proto_H['Vehicle'][1] \
                    and cls_proto_W['Vehicle'][0] < w <= cls_proto_W['Vehicle'][1]:
                if return_name:
                    ob_cls.append('Vehicle')
                else:
                    ob_cls.append(cls_proto['Vehicle'])
            elif cls_proto_L['Dis_Large'][0] < l <= cls_proto_L['Dis_Large'][1] \
                    and cls_proto_H['Dis_Large'][0] < h <= cls_proto_H['Dis_Large'][1] \
                    and cls_proto_W['Dis_Large'][0] < w <= cls_proto_W['Dis_Large'][1]:
                if return_name:
                    ob_cls.append('Dis_Large')
                else:
                    ob_cls.append(cls_proto['Dis_Large'])
            else:
                if return_name:
                    ob_cls.append('Dis_Small')
                else:
                    ob_cls.append(cls_proto['Dis_Small'])

        return np.array(boxes), np.array(ob_cls), np.array(ob_dif)

    def __call__(self, points):

        non_ground_points = self.remove_ground(points)
        clusters, labels = self.clustering(non_ground_points)
        boxes = self.box_fit(clusters)

        return boxes

class TrackSmooth:
    def __init__(self, config):
        self.tracker_config = config
        self.tracker = Tracker3D(box_type='OpenPCDet', config=self.tracker_config)

    def tracking(self, all_objects, all_pose, scores = None):

        self.all_pose = all_pose

        if scores is None:
            for i, boxes in enumerate(all_objects):
                self.tracker.tracking(boxes, scores=np.ones(shape=(len(boxes),)) * 100, timestamp=i,
                                      pose=all_pose[i])
        else:
            for i, boxes in enumerate(all_objects):
                self.tracker.tracking(boxes, scores=scores[i], timestamp=i,
                                      pose=all_pose[i])

        tracks = self.tracker.post_processing(self.tracker_config, self.all_pose)

        self.frame_first_dict = {}
        for ob_id in tracks.keys():
            track = tracks[ob_id]

            if track.last_updated_timestamp - track.first_updated_timestamp < self.tracker_config.remove_short_track:
                continue

            all_position = []
            all_speed = []

            for frame_id in track.trajectory.keys():

                ob = track.trajectory[frame_id]
                if ob.updated_state is None:
                    continue

                ob_state = np.array(ob.updated_state.T)
                all_position.append(ob_state[0, 0:3])
                all_speed.append(ob_state[0, 3:6])

            all_position = np.array(all_position)
            mean_position = np.mean(all_position[:, 0:2], 0)
            position_dis = all_position[:, 0:2] - mean_position
            dis = np.linalg.norm(position_dis, axis=1)
            std = np.std(dis)

            all_speed = np.array(all_speed)
            speed = np.linalg.norm(all_speed, axis=1)
            speed = np.mean(speed)

            for frame_id in track.trajectory.keys():

                ob = track.trajectory[frame_id]
                if ob.updated_state is None:
                    continue


                if frame_id in self.frame_first_dict.keys():
                    self.frame_first_dict[frame_id].append((ob_id, np.array(ob.updated_state.T), std, speed, ob.score))
                else:
                    self.frame_first_dict[frame_id] = [(ob_id, np.array(ob.updated_state.T), std, speed, ob.score)]

    def get_current_frame_objects_and_cls(self, frame_id, return_name = True):
        objects = []
        obj_ids = []
        ob_cls = []
        ob_dif = []

        if frame_id not in self.frame_first_dict:

            return np.empty(shape=(0, 7)), np.empty(shape=(0,)), np.empty(shape=(0,)), np.empty(shape=(0,))

        all_obs = self.frame_first_dict[frame_id]
        pose = self.all_pose[frame_id]

        cls_proto = self.tracker_config.cls

        cls_proto_L = self.tracker_config.cls_L

        cls_proto_W = self.tracker_config.cls_W

        cls_proto_H = self.tracker_config.cls_H

        for ob_id, ob_state, std, speed, score in all_obs:

            obj_ids.append(ob_id)

            box_template = np.zeros(shape=(1, 7))
            box_template[0, 0:3] = ob_state[0, 0:3]
            box_template[0, 3:7] = ob_state[0, 9:13]

            new_pose = np.mat(pose).I
            box = register_bbs(box_template, new_pose)

            objects.append(box)
            ob_dif.append(1)


            l, w, h = box_template[0, 3], box_template[0, 4], box_template[0, 5]
            top_z = box_template[0, 2]+h/2

            if top_z > self.tracker_config.max_top_z or w>self.tracker_config.max_width \
                    or l>self.tracker_config.max_len:
                if return_name:
                    ob_cls.append('Dis_Large')
                else:
                    ob_cls.append(cls_proto['Dis_Large'])
            elif cls_proto_L['Dis_Small'][0]<l<=cls_proto_L['Dis_Small'][1] \
                    and cls_proto_H['Dis_Small'][0]<h<=cls_proto_H['Dis_Small'][1]\
                    and cls_proto_W['Dis_Small'][0]<w<=cls_proto_W['Dis_Small'][1]:
                if return_name:
                    ob_cls.append('Dis_Small')
                else:
                    ob_cls.append(cls_proto['Dis_Small'])
            elif cls_proto_L['Pedestrian'][0]<l<=cls_proto_L['Pedestrian'][1] \
                    and cls_proto_H['Pedestrian'][0]<h<=cls_proto_H['Pedestrian'][1] \
                    and cls_proto_W['Pedestrian'][0] < w <= cls_proto_W['Pedestrian'][1]:
                if return_name:
                    ob_cls.append('Pedestrian')
                else:
                    ob_cls.append(cls_proto['Pedestrian'])
            elif cls_proto_L['Cyclist'][0]<l<=cls_proto_L['Cyclist'][1] \
                    and cls_proto_H['Cyclist'][0]<h<=cls_proto_H['Cyclist'][1] \
                    and cls_proto_W['Cyclist'][0] < w <= cls_proto_W['Cyclist'][1]:
                if return_name:
                    ob_cls.append('Cyclist')
                else:
                    ob_cls.append(cls_proto['Cyclist'])
            elif cls_proto_L['Vehicle'][0]<l<=cls_proto_L['Vehicle'][1] \
                    and cls_proto_H['Vehicle'][0]<h<=cls_proto_H['Vehicle'][1]\
                    and cls_proto_W['Vehicle'][0] < w <= cls_proto_W['Vehicle'][1]:
                if return_name:
                    ob_cls.append('Vehicle')
                else:
                    ob_cls.append(cls_proto['Vehicle'])
            elif cls_proto_L['Dis_Large'][0] < l <= cls_proto_L['Dis_Large'][1] \
                 and cls_proto_H['Dis_Large'][0] < h <= cls_proto_H['Dis_Large'][1]\
                 and cls_proto_W['Dis_Large'][0] < w <= cls_proto_W['Dis_Large'][1]:
                if return_name:
                    ob_cls.append('Dis_Large')
                else:
                    ob_cls.append(cls_proto['Dis_Large'])
            else:
                if return_name:
                    ob_cls.append('Dis_Small')
                else:
                    ob_cls.append(cls_proto['Dis_Small'])

        if len(objects)>0:
            objects=np.concatenate(objects)
            return objects, np.array(obj_ids), np.array(ob_cls), np.array(ob_dif)
        else:
            return np.empty(shape=(0, 7)), np.empty(shape=(0,)), np.empty(shape=(0,)), np.empty(shape=(0,))

    def get_current_frame_objects_and_cls_mv(self, frame_id, return_name = True):
        objects = []
        obj_ids = []
        ob_cls = []
        ob_dif = []
        ob_std = []
        ob_speed = []
        ob_score = []

        if frame_id not in self.frame_first_dict:
            return np.empty(shape=(0, 7)), \
                   np.empty(shape=(0,)), \
                   np.empty(shape=(0,)), \
                   np.empty(shape=(0,)), \
                   np.empty(shape=(0,)), \
                   np.empty(shape=(0,)), \
                   np.empty(shape=(0,))

        all_obs = self.frame_first_dict[frame_id]
        pose = self.all_pose[frame_id]

        cls_proto = self.tracker_config.cls

        cls_proto_L = self.tracker_config.cls_L

        cls_proto_W = self.tracker_config.cls_W

        cls_proto_H = self.tracker_config.cls_H

        for ob_id, ob_state, std, speed, score in all_obs:

            obj_ids.append(ob_id)

            box_template = np.zeros(shape=(1, 7))
            box_template[0, 0:3] = ob_state[0, 0:3]
            box_template[0, 3:7] = ob_state[0, 9:13]

            new_pose = np.mat(pose).I
            box = register_bbs(box_template, new_pose)

            objects.append(box)
            ob_dif.append(1)
            ob_std.append(std)
            ob_speed.append(speed)
            ob_score.append(score)


            l, w, h = box_template[0, 3], box_template[0, 4], box_template[0, 5]
            top_z = box_template[0, 2]+h/2

            if top_z > self.tracker_config.max_top_z or w>self.tracker_config.max_width \
                    or l>self.tracker_config.max_len:
                if return_name:
                    ob_cls.append('Dis_Large')
                else:
                    ob_cls.append(cls_proto['Dis_Large'])
            elif cls_proto_L['Dis_Small'][0]<l<=cls_proto_L['Dis_Small'][1] \
                    and cls_proto_H['Dis_Small'][0]<h<=cls_proto_H['Dis_Small'][1]\
                    and cls_proto_W['Dis_Small'][0]<w<=cls_proto_W['Dis_Small'][1]:
                if return_name:
                    ob_cls.append('Dis_Small')
                else:
                    ob_cls.append(cls_proto['Dis_Small'])
            elif cls_proto_L['Pedestrian'][0]<l<=cls_proto_L['Pedestrian'][1] \
                    and cls_proto_H['Pedestrian'][0]<h<=cls_proto_H['Pedestrian'][1] \
                    and cls_proto_W['Pedestrian'][0] < w <= cls_proto_W['Pedestrian'][1]:
                if return_name:
                    ob_cls.append('Pedestrian')
                else:
                    ob_cls.append(cls_proto['Pedestrian'])
            elif cls_proto_L['Cyclist'][0]<l<=cls_proto_L['Cyclist'][1] \
                    and cls_proto_H['Cyclist'][0]<h<=cls_proto_H['Cyclist'][1] \
                    and cls_proto_W['Cyclist'][0] < w <= cls_proto_W['Cyclist'][1]:
                if return_name:
                    ob_cls.append('Cyclist')
                else:
                    ob_cls.append(cls_proto['Cyclist'])
            elif cls_proto_L['Vehicle'][0]<l<=cls_proto_L['Vehicle'][1] \
                    and cls_proto_H['Vehicle'][0]<h<=cls_proto_H['Vehicle'][1]\
                    and cls_proto_W['Vehicle'][0] < w <= cls_proto_W['Vehicle'][1]:
                if return_name:
                    ob_cls.append('Vehicle')
                else:
                    ob_cls.append(cls_proto['Vehicle'])
            elif cls_proto_L['Dis_Large'][0] < l <= cls_proto_L['Dis_Large'][1] \
                 and cls_proto_H['Dis_Large'][0] < h <= cls_proto_H['Dis_Large'][1]\
                 and cls_proto_W['Dis_Large'][0] < w <= cls_proto_W['Dis_Large'][1]:
                if return_name:
                    ob_cls.append('Dis_Large')
                else:
                    ob_cls.append(cls_proto['Dis_Large'])
            else:
                if return_name:
                    ob_cls.append('Dis_Small')
                else:
                    ob_cls.append(cls_proto['Dis_Small'])

        if len(objects)>0:
            objects=np.concatenate(objects)
            return objects, \
                   np.array(obj_ids), \
                   np.array(ob_cls),\
                   np.array(ob_dif),\
                   np.array(ob_std),\
                   np.array(ob_speed),\
                   np.array(ob_score)
        else:
            return np.empty(shape=(0, 7)), \
                   np.empty(shape=(0,)), \
                   np.empty(shape=(0,)), \
                   np.empty(shape=(0,)), \
                   np.empty(shape=(0,)), \
                   np.empty(shape=(0,)), \
                   np.empty(shape=(0,))


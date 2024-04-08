import numpy as np
from .object import Object

class Trajectory:
    def __init__(self,init_bb=None,
                 init_features=None,
                 init_score=None,
                 init_timestamp=None,
                 label=None,
                 tracking_features=True,
                 bb_as_features=False,
                 config = None
                 ):
        """

        Args:
            init_bb: array(7) or array(7*k), 3d box or tracklet
            init_features: array(m), features of box or tracklet
            init_score: array(1) or float, score of detection
            init_timestamp: int, init timestamp
            label: int, unique ID for this trajectory
            tracking_features: bool, if track features
            bb_as_features: bool, if treat the bb as features
        """
        assert init_bb is not None

        self.init_bb = init_bb
        self.init_features = init_features
        self.init_score = init_score
        self.init_timestamp = init_timestamp
        self.label = label
        self.tracking_bb_size = True
        self.tracking_features = tracking_features
        self.bb_as_features = bb_as_features

        self.config = config


        self.scanning_interval = 1./self.config.LiDAR_scanning_frequency

        if self.bb_as_features:
            if self.init_features is None:
                self.init_features = init_bb
            else:
                self.init_features = np.concatenate([init_bb,init_features],0)
        self.trajectory = {}

        self.track_dim = self.compute_track_dim() # 9+4+bb_features.shape

        self.init_parameters()
        self.init_trajectory()


        self.consecutive_missed_num = 0
        self.first_updated_timestamp = init_timestamp
        self.last_updated_timestamp = init_timestamp

    def __len__(self):
        return len(self.trajectory)

    def compute_track_dim(self):
        """
        compute tracking dimension
        :return:
        """
        track_dim=9 #x,y,z,vx,vy,vz,ax,ay,az

        if self.tracking_bb_size:
            track_dim+=4 # w,h,l,yaw
        if self.tracking_features:
            track_dim+=self.init_features.shape[0] #features.shape
        return track_dim

    def init_trajectory(self):
        """
        first initialize the object state with the input boxes info,
        then initialize the trajectory with the initialized object.
        :return:
        """

        detected_state_template = np.zeros(shape=(self.track_dim-6))

        update_covariance_template = np.eye(self.track_dim)*0.01

        detected_state_template[:3] = self.init_bb[:3] #init x,y,z

        if self.tracking_bb_size:
            detected_state_template[3: 7] = self.init_bb[3:7]
            if self.tracking_features:
                detected_state_template[7: ] = self.init_features[:]
        else:
            if self.tracking_features:

                detected_state_template[3: ] = self.init_features[:]

        detected_state_template = np.mat(detected_state_template).T
        update_covariance_template = np.mat(update_covariance_template).T

        update_state_template = self.H * detected_state_template

        object = Object()

        object.updated_state = update_state_template
        object.predicted_state = update_state_template
        object.detected_state = detected_state_template
        object.updated_covariance =update_covariance_template
        object.predicted_covariance = update_covariance_template
        object.prediction_score = 1
        object.score=self.init_score
        object.features = self.init_features

        self.trajectory[self.init_timestamp] = object

    def init_parameters(self):
        """
        initialize KF tracking parameters
        :return:
        """
        self.A = np.mat(np.eye(self.track_dim))
        self.Q = np.mat(np.eye(self.track_dim))*self.config.state_func_covariance
        self.P = np.mat(np.eye(self.track_dim-6))*self.config.measure_func_covariance
        self.B = np.mat(np.zeros(shape=(self.track_dim-6,self.track_dim)))
        self.B[0:3,:] = self.A[0:3,:]
        self.B[3:,:] = self.A[9:,:]

        self.velo = np.mat(np.eye(3))*self.scanning_interval
        self.acce = np.mat(np.eye(3))*0.5*self.scanning_interval**2

        self.A[0:3,3:6] = self.velo
        self.A[3:6,6:9] = self.velo
        self.A[0:3,6:9] = self.acce

        self.H = self.B.T
        self.K = np.mat(np.zeros(shape=(self.track_dim,self.track_dim)))
        self.K[3, 0] = self.scanning_interval
        self.K[4, 1] = self.scanning_interval
        self.K[5, 2] = self.scanning_interval

    def state_prediction(self,timestamp):
        """
        predict the object state at the given timestamp
        """

        previous_timestamp = timestamp-1

        assert previous_timestamp in self.trajectory.keys()

        previous_object = self.trajectory[previous_timestamp]

        if previous_object.updated_state is not None:
            previous_state = previous_object.updated_state
            previous_covariance = previous_object.updated_covariance
        else:
            previous_state = previous_object.predicted_state
            previous_covariance = previous_object.predicted_covariance

        previous_prediction_score = previous_object.prediction_score

        if timestamp-1 in self.trajectory.keys():
            if self.trajectory[timestamp-1].updated_state is not None:
                current_prediction_score = previous_prediction_score * (1 - self.config.prediction_score_decay*15)
            else:
                current_prediction_score = previous_prediction_score * (1 - self.config.prediction_score_decay)
        else:
            current_prediction_score = previous_prediction_score * (1 - self.config.prediction_score_decay)


        current_predicted_state = self.A*previous_state
        current_predicted_covariance = self.A*previous_covariance*self.A.T + self.Q

        new_ob = Object()

        new_ob.predicted_state = current_predicted_state
        new_ob.predicted_covariance = current_predicted_covariance
        new_ob.prediction_score = current_prediction_score

        self.trajectory[timestamp] = new_ob
        self.consecutive_missed_num += 1

    def sigmoid(self,x):
        return 1.0/(1+np.exp(-float(x)))

    def state_update(self,
                     bb=None,
                     features=None,
                     score=None,
                     timestamp=None,
                     ):
        """
        update the trajectory
        Args:
            bb: array(7) or array(7*k), 3D box or tracklet
            features: array(m), features of box or tracklet
            score:
            timestamp:
        """
        assert bb is not None
        assert timestamp in self.trajectory.keys()

        if self.bb_as_features:
            if features is None:
                features = bb
            else:
                features = np.concatenate([bb,features],0)

        detected_state_template = np.zeros(shape=(self.track_dim-6))

        detected_state_template[:3] = bb[:3] #init x,y,z

        if self.tracking_bb_size:
            detected_state_template[3: 7] = bb[3:7]
            if self.tracking_features:
                detected_state_template[7: ] = features[:]
        else:
            if self.tracking_features:
                detected_state_template[3: ] = features[:]

        detected_state_template = np.mat(detected_state_template).T

        current_ob = self.trajectory[timestamp]

        predicted_state = current_ob.predicted_state
        predicted_covariance = current_ob.predicted_covariance

        temp = self.B*predicted_covariance*self.B.T+self.P

        KF_gain = predicted_covariance*self.B.T*temp.I

        updated_state = predicted_state+KF_gain*(detected_state_template-self.B*predicted_state)
        updated_covariance = (np.mat(np.eye(self.track_dim)) - KF_gain*self.B)*predicted_covariance

        if len(self.trajectory)==2:

            updated_state = self.H*detected_state_template+\
                            self.K*(self.H*detected_state_template-self.trajectory[timestamp-1].updated_state)


        current_ob.updated_state = updated_state
        current_ob.updated_covariance = updated_covariance
        current_ob.detected_state = detected_state_template

        if self.consecutive_missed_num>1:
            current_ob.prediction_score = 1 # using one to update the score is enough
        elif self.trajectory[timestamp - 1].updated_state is not None:
            current_ob.prediction_score = current_ob.prediction_score + self.config.prediction_score_decay*15*(self.sigmoid(score))
        else:
            current_ob.prediction_score = current_ob.prediction_score + self.config.prediction_score_decay*(self.sigmoid(score))
        current_ob.score = score
        current_ob.features = features

        self.consecutive_missed_num = 0
        self.last_updated_timestamp = timestamp

    def limit(self, ang):
        ang = ang % (2 * np.pi)

        ang[ang > np.pi] = ang[ang > np.pi] - 2 * np.pi

        ang[ang < -np.pi] = ang[ang < -np.pi] + 2 * np.pi

        return ang

    def points_rigid_transform(self, cloud, pose):
        if cloud.shape[0] == 0:
            return cloud
        mat = np.ones(shape=(cloud.shape[0], 4), dtype=np.float32)
        pose_mat = np.mat(pose)
        mat[:, 0:3] = cloud[:, 0:3]
        mat = np.mat(mat)
        transformed_mat = pose_mat * mat.T
        T = np.array(transformed_mat.T, dtype=np.float32)
        return T[:, 0:3]

    def filtering2(self, config, pose = None):
        """
        filtering the trajectory in a global or near online way
        """

        wind_size = int(config.LiDAR_scanning_frequency*config.latency)

        if wind_size <0:

            detected_num = 0.00001
            score_sum = 0

            ob_num = 0.000001

            all_l, all_w, all_h, all_ya = {}, {}, {}, {}

            for key in self.trajectory.keys():
                ob = self.trajectory[key]
                if ob.score is not None:
                    detected_num+=1
                    score_sum+=ob.score

                if self.first_updated_timestamp<=key<=self.last_updated_timestamp and ob.updated_state is None:
                    ob.updated_state = ob.predicted_state

                if ob.updated_state is not None:

                    if ob.updated_state[9, 0]<ob.updated_state[10, 0]:
                        temp = ob.updated_state[9, 0]
                        ob.updated_state[9, 0] = ob.updated_state[10, 0]
                        ob.updated_state[10, 0] = temp
                        ob.updated_state[12, 0] += np.pi/2
                    ob_num+=1

                    l = ob.updated_state[9, 0]
                    w = ob.updated_state[10, 0]
                    h = ob.updated_state[11, 0]
                    ya = ob.updated_state[12, 0]

                    all_l[key] = l
                    all_w[key] = w
                    all_h[key] = h
                    all_ya[key] = ya

            mean_lwh_size = 10
            mean_ya_size = 10

            for key in self.trajectory.keys():
                sub_all_l, sub_all_w, sub_all_h, sub_all_ya = [], [], [], []
                for w_i in range(key-mean_lwh_size, key+mean_lwh_size):
                    if w_i in all_l.keys():
                        sub_all_l.append(all_l[w_i])
                        sub_all_w.append(all_w[w_i])
                        sub_all_h.append(all_h[w_i])

                for w_i in range(key-mean_ya_size, key+mean_ya_size):
                    if w_i in all_l.keys():
                        sub_all_ya.append(all_ya[w_i])

                sub_ob = self.trajectory[key]
                if sub_ob.updated_state is not None:
                    new_ya = sub_ob.updated_state[12, 0]

                    angles = np.array(sub_all_ya)
                    angles = self.limit(angles)
                    res = angles - new_ya
                    res = self.limit(res)
                    res = res[np.abs(res) < 2]
                    res = res.mean()
                    b = new_ya + res
                    sub_ob.updated_state[12, 0] = b

                    sub_ob.updated_state[9, 0] = np.mean(sub_all_l)
                    sub_ob.updated_state[10, 0] = np.mean(sub_all_w)
                    sub_ob.updated_state[11, 0] = np.mean(sub_all_h)


        else:
            keys = list(self.trajectory.keys())

            for key in keys:

                min_key = int(key-wind_size)
                max_key = int(key+wind_size)
                detected_num = 0.00001
                score_sum = 0
                for key_i in range(min_key,max_key):
                    if key_i not in self.trajectory:
                        continue
                    ob = self.trajectory[key_i]
                    if ob.score is not None:
                        detected_num+=1
                        score_sum+=ob.score
                    if self.first_updated_timestamp<=key_i<=self.last_updated_timestamp and ob.updated_state is None:
                        ob.updated_state = ob.predicted_state

                score = score_sum / detected_num
                if wind_size!=0:
                    self.trajectory[key].score=score

    def softmax(self, x):
        x_row_max = x.max(axis=-1)
        x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
        x = x - x_row_max
        x_exp = np.exp(x)
        x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
        softmax = x_exp / x_exp_row_sum
        return softmax


    def filtering(self, config, pose = None):
        """
        filtering the trajectory in a global or near online way
        """

        wind_size = int(config.LiDAR_scanning_frequency*config.latency)

        if wind_size <0:

            detected_num = 0.00001
            score_sum = 0

            ob_num = 0.000001

            all_scores = []

            all_l, all_w, all_h, all_ya = {}, {}, {}, {}
            all_dis = {}

            for key in self.trajectory.keys():
                ob = self.trajectory[key]
                if ob.score is not None:
                    detected_num+=1
                    score_sum+=ob.score
                    all_scores.append(ob.score)

                if self.first_updated_timestamp <= key <= self.last_updated_timestamp and ob.updated_state is None:

                    near_left = None
                    near_left_counter = 0
                    near_left_key = key - 1

                    while (near_left is None) and near_left_key > key - config.max_prediction_num:
                        near_left = self.trajectory[near_left_key].updated_state
                        near_left_counter += 1
                        near_left_key-=1

                    near_right = None
                    near_right_counter = 0
                    near_right_key = key + 1

                    while (near_right is None) and near_right_key < key + config.max_prediction_num:
                        near_right = self.trajectory[near_right_key].updated_state
                        near_right_counter += 1
                        near_right_key+=1

                    if (near_left is not None) and (near_right is not None):
                        sums = (near_left_counter+near_right_counter)
                        new_x = (near_left_counter/sums) * near_left[0,0]+\
                                (near_right_counter/sums) * near_right[0,0]

                        new_y = (near_left_counter /sums) * near_left[1, 0] + \
                                (near_right_counter /sums) * near_right[1, 0]

                        new_z = (near_left_counter /sums) * near_left[2, 0] + \
                                (near_right_counter /sums) * near_right[2, 0]

                        ob.predicted_state[0, 0] = new_x
                        ob.predicted_state[1, 0] = new_y
                        ob.predicted_state[2, 0] = new_z


                if self.first_updated_timestamp<=key<=self.last_updated_timestamp and ob.updated_state is None:

                    ob.updated_state = ob.predicted_state


                if ob.updated_state is not None:

                    if ob.updated_state[9, 0]<ob.updated_state[10, 0]:
                        temp = ob.updated_state[9, 0]
                        ob.updated_state[9, 0] = ob.updated_state[10, 0]
                        ob.updated_state[10, 0] = temp
                        ob.updated_state[12, 0] += np.pi/2
                    ob_num+=1

                    l = ob.updated_state[9, 0]
                    w = ob.updated_state[10, 0]
                    h = ob.updated_state[11, 0]
                    ya = ob.updated_state[12, 0]

                    this_dis = np.array([[ob.updated_state[0, 0],ob.updated_state[1, 0],ob.updated_state[2, 0]]])

                    this_dis = self.points_rigid_transform(this_dis, np.linalg.inv(pose[key]))

                    all_dis[key] = np.linalg.norm(this_dis)

                    all_l[key] = l
                    all_w[key] = w
                    all_h[key] = h
                    all_ya[key] = ya

            mean_lwh_size = config.lwh_win_size
            mean_ya_size = config.yaw_win_size

            for key in self.trajectory.keys():
                sub_all_l, sub_all_w, sub_all_h, sub_all_ya, sub_all_dis = [], [], [], [], []
                for w_i in range(key-mean_lwh_size, key+mean_lwh_size):
                    if w_i in all_l.keys():
                        sub_all_l.append(all_l[w_i])
                        sub_all_w.append(all_w[w_i])
                        sub_all_h.append(all_h[w_i])
                        sub_all_dis.append(all_dis[w_i])

                for w_i in range(key-mean_ya_size, key+mean_ya_size):
                    if w_i in all_l.keys():
                        sub_all_ya.append(all_ya[w_i])

                sub_ob = self.trajectory[key]
                if sub_ob.updated_state is not None and mean_lwh_size>0:
                    sub_all_dis = np.array([sub_all_dis])
                    sub_all_dis -= sub_all_dis.min()
                    sub_all_dis /= sub_all_dis.max() + 0.1
                    sub_all_dis = 1 - sub_all_dis + 0.1
                    weights = self.softmax(sub_all_dis)[0]

                    new_ya = sub_ob.updated_state[12, 0]

                    angles = np.array(sub_all_ya)
                    angles = self.limit(angles)
                    res = angles - new_ya
                    res = self.limit(res)
                    res = res[np.abs(res) < 2]

                    res = res.mean()

                    b = new_ya + res
                    sub_ob.updated_state[12, 0] = b


                    sub_ob.updated_state[9, 0] = np.sum(np.array(sub_all_l)*weights)
                    sub_ob.updated_state[10, 0] = np.sum(np.array(sub_all_w)*weights)
                    sub_ob.updated_state[11, 0] = np.sum(np.array(sub_all_h)*weights)

                sub_ob.score = np.mean(all_scores)


        else:
            keys = list(self.trajectory.keys())

            for key in keys:

                min_key = int(key-wind_size)
                max_key = int(key+wind_size)
                detected_num = 0.00001
                score_sum = 0
                for key_i in range(min_key,max_key):
                    if key_i not in self.trajectory:
                        continue
                    ob = self.trajectory[key_i]
                    if ob.score is not None:
                        detected_num+=1
                        score_sum+=ob.score
                    if self.first_updated_timestamp<=key_i<=self.last_updated_timestamp and ob.updated_state is None:
                        ob.updated_state = ob.predicted_state

                score = score_sum / detected_num
                if wind_size!=0:
                    self.trajectory[key].score=score
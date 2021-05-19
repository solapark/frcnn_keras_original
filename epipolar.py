import numpy as np
from scipy.spatial.transform import Rotation as R

class EPIPOLAR :
    def __init__(self, args):
        self.intrin = args.intrin #(num_valid_cam, 3, 3)
        self.diag = np.sqrt(args.width**2 + args.height**2)
        self.num_valid_cam = args.num_valid_cam

    def ext_a2b(self, ext_a, ext_b):
        T_a2r = np.eye(4)
        T_a2r[0:3, 0:3] = R.from_euler('xyz', ext_a[3:]).as_dcm()
        T_a2r[0:3, 3] = np.array(ext_a[:3])

        T_b2r = np.eye(4)
        T_b2r[0:3, 0:3] = R.from_euler('xyz', ext_b[3:]).as_dcm()
        T_b2r[0:3, 3] = np.array(ext_b[:3])

        # T_a2b = T_r2b * T_a2r = T_b2r.inv * T_a2r
        T_a2b = np.matmul(np.linalg.inv(T_b2r), T_a2r)

        return T_a2b

    def find_line(self, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        d = (y2 - y1) / (x2 - x1)
        e = y1 - x1 * d
        return [-d, 1, -e]

    def find_foot(self, a, b, c, pt):
        x1, y1 = pt
        temp = (-1 * (a * x1 + b * y1 + c) / (a * a + b * b))
        x = temp * a + x1
        y = temp * b + y1
        return [x, y]

    def find_dist(self, pt1, pt2):
        return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5

    def calc_T_a2b(self, extrin):
        #input : extrin #(num_valid_cam, 6)
        self.T_a2b = np.zeros((self.num_valid_cam, self.num_valid_cam, 4, 4))
        for i in range(self.num_valid_cam):
            for j in range(self.num_valid_cam):
                if(i == j): continue
                self.T_a2b[i, j] = self.ext_a2b(extrin[i], extrin[j])
        
    def get_epipolar_dist(self, cam1_idx, cam2_idx, bbox_list1, bbox_list2):
        '''
        inputs:
            cam1_idx #int
            cam2_idx #int
            bbox list1 #(N, 4)
            bbox list2 #(M, 4)
        outputs :
            dist #(N, M)
                dist[i, j] = distance between jth box in cam2 and epipolar line on cam2 drawn by ith box in cam1 
        '''
        #T_a2b = self.ext_a2b(extrin1, extrin2)
        T_a2b = self.T_a2b[cam1_idx, cam2_idx]

        dist_matrix = np.zeros((len(bbox_list1), len(bbox_list2)))

        for i in range(len(bbox_list1)):
            for j in range(len(bbox_list2)):
                b1x1, b1y1, b1x2, b1y2 = bbox_list1[i]
                b2x1, b2y1, b2x2, b2y2 = bbox_list2[j]
                bbox1_2dpt = ((b1x1 + b1x2) / 2, (b1y1 + b1y2) / 2)
                bbox2_2dpt = ((b2x1 + b2x2) / 2, (b2y1 + b2y2) / 2)

                # bbox 1 in camera 2
                bbox1_3dpt = np.matmul(np.linalg.inv(self.intrin[cam1_idx]), np.array([*bbox1_2dpt, 1]))
                bbox1_3dpt = np.array([*bbox1_3dpt.tolist(), 1])

                bbox1_in2_3dpt = np.matmul(T_a2b, bbox1_3dpt)[:3]
                bbox1_in2_2dpt = np.matmul(self.intrin[cam2_idx], bbox1_in2_3dpt)
                bbox1_in2_2dpt = bbox1_in2_2dpt[:2] / bbox1_in2_2dpt[2]

                # camera 1 epipole in camera 2
                epipole1_3dpt = np.array([0, 0, 0, 1])
                epipole1_in2_3dpt = np.matmul(T_a2b, epipole1_3dpt)[:3]
                epipole1_in2_2dpt = np.matmul(self.intrin[cam2_idx], epipole1_in2_3dpt)
                epipole1_in2_2dpt = epipole1_in2_2dpt[:2] / epipole1_in2_2dpt[2]

                # find epipolar line
                a, b, c = self.find_line(bbox1_in2_2dpt, epipole1_in2_2dpt)

                foot = self.find_foot(a, b, c, bbox2_2dpt)
                dist = self.find_dist(bbox2_2dpt, foot)

                # measure distance
                dist_matrix[i, j] = dist

        # normalize by diagonal line
        dist_matrix = dist_matrix / self.diag
        return dist_matrix

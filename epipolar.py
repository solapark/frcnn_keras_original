import numpy as np
from scipy.spatial.transform import Rotation as R
from sympy import symbols
import utility
import cv2

class EPIPOLAR :
    def __init__(self, args):
        self.intrin = args.intrin #(num_valid_cam, 3, 3)
        self.num_valid_cam = args.num_valid_cam

        self.rpn_stride = args.rpn_stride

        self.width = args.width
        self.height = args.height

        self.zoom_out_w = args.rpn_stride * self.width/args.resized_width
        self.zoom_out_h = args.rpn_stride * self.height/args.resized_height

        self.diag = np.sqrt(args.width**2 + args.height**2)

    def reset(self, extrins, debug_imgs) :
        self.calc_T_a2b(extrins)
        self.calc_epipole()
        self.debug_imgs = debug_imgs

    def resized_box_to_original_box(self, bbox_list):
        x1 = bbox_list[:, 0] * self.zoom_out_w
        y1 = bbox_list[:, 1] * self.zoom_out_h
        x2 = bbox_list[:, 2] * self.zoom_out_w
        y2 = bbox_list[:, 3] * self.zoom_out_h
        return np.column_stack([x1, y1, x2, y2]).astype('int32')

    def draw_result(self, cam1_idx, cam2_idx, box1, box2, foot, a, b, c):
        src_img = cv2.resize(self.debug_imgs[cam1_idx], (self.width, self.height)) 
        dst_img = cv2.resize(self.debug_imgs[cam2_idx], (self.width, self.height))
        src_reuslt_img = utility.draw_box(src_img, box1, name = None, color = (0, 0, 255), is_show = False)
        dst_reuslt_img = utility.draw_box(dst_img, box2, name = None, color = (0, 0, 255), is_show = False)
        line_start = (0, int(-c/b))
        line_end = (int(self.width), int(-a*self.width/b - c))
        print('line', line_start, line_end)
        dst_reuslt_img = utility.draw_line(dst_reuslt_img, line_start, line_end)
        dst_reuslt_img = cv2.circle(dst_reuslt_img, tuple(map(int, foot)), 5, (0, 0, 255), -1)

        img_list = [src_reuslt_img, dst_reuslt_img]
        concat_img = utility.get_concat_img(img_list, cols=2)
        resized_concat_img = cv2.resize(concat_img, (640, 360))
        cv2.imshow('epipolar', resized_concat_img)
        cv2.waitKey(0)

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
        
    def calc_epipole(self) :
        self.epipole = np.zeros((self.num_valid_cam, self.num_valid_cam, 2))
        epipole1_3dpt = np.array([0, 0, 0, 1])
        for i in range(self.num_valid_cam):
            for j in range(self.num_valid_cam) :
                if(i==j) : continue
                epipole1_in2_3dpt = np.matmul(self.T_a2b[i, j], epipole1_3dpt)[:3]
                epipole1_in2_2dpt = np.matmul(self.intrin[j], epipole1_in2_3dpt)
                self.epipole[i, j] = epipole1_in2_2dpt[:2] / epipole1_in2_2dpt[2]


    def get_box_ids_on_epiline(self, ref_cam_idx, target_cam_idx, ref_box, target_boxes) :
        ref_box = ref_box.reshape(-1, 4)
        dist = self.get_epipolar_dist(ref_cam_idx, target_cam_idx, ref_box, target_boxes)
        valid_idx = np.where(dist < self.max_dist_to_epipolar_line)[1]
        return valid_idx

    def get_epipolar_line_cross_pnt(ref_cam1_idx, ref_box1, ref_cam2_idx, ref_box2, target_cam_idx) :

        ref1_T_a2b = self.T_a2b[ref_cam1_idx, target_cam_idx]
        ref2_T_a2b = self.T_a2b[ref_cam2_idx, target_cam_idx]
        ref1_epipole1_in2_2dpt = self.epipole[ref_cam1_idx, target_cam_idx]
        ref2_epipole1_in2_2dpt = self.epipole[ref_cam2_idx, target_cam_idx]

        ref1_intrin = self.intrin[ref_cam1_idx]
        ref2_intrin = self.intrin[ref_cam2_idx]
        target_intrin = self.intrin[target_cam_idx]

        epipolar_line1 = get_epipolar_line(self, ref1_T_a2b, ref1_epipole1_in2_2dpt, ref1_intrin, target_intrin, ref_box1)
        epipolar_line2 = get_epipolar_line(self, ref2_T_a2b, ref2_epipole1_in2_2dpt, ref2_intrin, target_intrin, ref_box2)

        cross_pnt = self.solve_system_of_equations(epipolar_line1, epipolar_line2)
        return cross_pnt

    def solve_system_of_equations(self, eq1, eq2):
        a1, b1, c1 = eq1
        a2, b2, c2 = eq2
        y = symbol('y')
        equation1 = a1*x + b1*y + c1
        equation2 = a2*x + b2*y + c2
        ans = solve((equation1, equation2), dict=True)
        #[{y: 11/7, x: -1/7}]
        return ans['x'], ans['y']

    def get_epipolar_line(self, T_a2b, epipole1_in2_2dpt, cam1_intrin, cam2_intrin, cam1_box): 
        x1, y1, x2, y2 = cam1_box
        bbox1_2dpt = (x1 + x2) / 2, (y1 + y2) / 2

        # bbox 1 in camera 2
        bbox1_3dpt = np.matmul(np.linalg.inv(cam1_intrin), np.array([*bbox1_2dpt, 1]))
        bbox1_3dpt = np.array([*bbox1_3dpt.tolist(), 1])

        bbox1_in2_3dpt = np.matmul(T_a2b, bbox1_3dpt)[:3]
        bbox1_in2_2dpt = np.matmul(self.cam2_intrin, bbox1_in2_3dpt)
        bbox1_in2_2dpt = bbox1_in2_2dpt[:2] / bbox1_in2_2dpt[2]

        # find epipolar line
        a, b, c = self.find_line(bbox1_in2_2dpt, epipole1_in2_2dpt)

        return a, b, c

    def get_epipolar_dist(self, cam1_idx, cam2_idx, bbox_list1_rpn, bbox_list2_rpn):
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
        T_a2b = self.T_a2b[cam1_idx, cam2_idx]

        # camera 1 epipole in camera 2
        epipole1_in2_2dpt = self.epipole[cam1_idx, cam2_idx]
        cam1_intrin = self.intrin[cam1_idx]
        cam2_intrin = self.intrin[cam2_idx]

        bbox_list1 = self.resized_box_to_original_box(bbox_list1_rpn)
        bbox_list2 = self.resized_box_to_original_box(bbox_list2_rpn)

        dist_matrix = np.zeros((len(bbox_list1), len(bbox_list2)))

        for i in range(len(bbox_list1)):
            '''
            b1x1, b1y1, b1x2, b1y2 = bbox_list1[i]
            bbox1_2dpt = ((b1x1 + b1x2) / 2, (b1y1 + b1y2) / 2)

            # bbox 1 in camera 2
            bbox1_3dpt = np.matmul(np.linalg.inv(self.intrin[cam1_idx]), np.array([*bbox1_2dpt, 1]))
            bbox1_3dpt = np.array([*bbox1_3dpt.tolist(), 1])

            bbox1_in2_3dpt = np.matmul(T_a2b, bbox1_3dpt)[:3]
            bbox1_in2_2dpt = np.matmul(self.intrin[cam2_idx], bbox1_in2_3dpt)
            bbox1_in2_2dpt = bbox1_in2_2dpt[:2] / bbox1_in2_2dpt[2]

            # find epipolar line
            a, b, c = self.find_line(bbox1_in2_2dpt, epipole1_in2_2dpt)
            '''
            a, b, c = self.get_epipolar_line(T_a2b, epipole1_in2_2dpt, cam1_intrin, cam2_intrin, bbox_list1[i])

            for j in range(len(bbox_list2)):
                b2x1, b2y1, b2x2, b2y2 = bbox_list2[j]
                bbox2_2dpt = ((b2x1 + b2x2) / 2, (b2y1 + b2y2) / 2)

                foot = self.find_foot(a, b, c, bbox2_2dpt)
                dist = self.find_dist(bbox2_2dpt, foot)

                # measure distance
                dist_matrix[i, j] = dist
                '''
                print('epi pole', -c/b, -c/a)
                print('dist', dist/self.diag)
                print('foot', foot)

                self.draw_result(cam1_idx, cam2_idx, bbox_list1[i], bbox_list2[j], foot, a, b, c)
                '''

        # normalize by diagonal line
        dist_matrix = dist_matrix / self.diag
        return dist_matrix

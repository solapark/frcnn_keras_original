import numpy as np
from scipy.spatial.transform import Rotation as R
from sympy import symbols
import utility
import cv2
import json

class EPIPOLAR :
    def __init__(self, args):
        self.num_cam = args.num_valid_cam
        self.max_dist_epiline_to_box = args.max_dist_epiline_to_box
        self.max_dist_epiline_cross_to_box = args.max_dist_epiline_cross_to_box
        self.num_valid_cam = args.num_valid_cam

        self.rpn_stride = args.rpn_stride

        self.width = args.width
        self.height = args.height

        self.zoom_out_w = args.rpn_stride * self.width/args.resized_width
        self.zoom_out_h = args.rpn_stride * self.height/args.resized_height

        self.diag = np.sqrt(args.width**2 + args.height**2)

        self.intrin = self.parse_intrin(args.dataset_path) #(num_valid_cam, 3, 3)

        self.view_x_margin = 30
        self.view_y_margin = 30

        self.args = args

    def parse_intrin(self, path):
        with open(path) as fp: 
            j = json.load(fp)

        intrins = j['intrinsics']
        intrins = [np.array(arr).reshape(3, 3) for arr in intrins.values()]
        return np.array(intrins)

    def reset(self, extrins, debug_imgs) :
        self.calc_T_a2b(extrins)
        self.calc_epipole()
        self.debug_imgs = debug_imgs

    def original_pnt_to_resized_pnt(self, pnt):
        x, y = pnt
        x /= self.zoom_out_w
        y /= self.zoom_out_h
        return np.array([x, y])

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
        print('box', box1, 'box', box2, 'line', line_start, line_end)
        dst_reuslt_img = utility.draw_line(dst_reuslt_img, line_start, line_end)
        dst_reuslt_img = cv2.circle(dst_reuslt_img, tuple(map(int, foot)), 5, (0, 0, 255), -1)

        img_list = [src_reuslt_img, dst_reuslt_img]
        concat_img = utility.get_concat_img(img_list, cols=2)
        resized_concat_img = cv2.resize(concat_img, (640, 360))
        cv2.imwrite('epipolar.jpg', resized_concat_img)
        #cv2.waitKey(0)

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

    def get_epipolar_lines(self, ref_cam_idx, ref_box):
        epipolar_line_dict = {}
        for offset in range(1, self.num_cam):
            target_cam_idx = (ref_cam_idx + offset) % self.num_cam
            epipolar_line_dict[target_cam_idx] = self.get_epipolar_line(ref_cam_idx, ref_box, target_cam_idx)

            '''
            a, b, c = self.get_epipolar_line(ref_cam_idx, ref_box, target_cam_idx)
            original_ref_box = self.resized_box_to_original_box(ref_box.reshape(-1, 4))
            ref_box_center = self.get_boxes_centers(original_ref_box)
            foot = self.find_foot(a, b, c, ref_box_center[0])
            self.draw_result(ref_cam_idx, target_cam_idx, original_ref_box[0], original_ref_box[0], foot, a, b, c)
            '''

        return epipolar_line_dict

    def get_epipolar_line(self, ref_cam_idx, ref_box, target_cam_idx):
        ref_T_a2b = self.T_a2b[ref_cam_idx, target_cam_idx]
        ref_epipole_in2_2dpt = self.epipole[ref_cam_idx, target_cam_idx]

        ref_intrin = self.intrin[ref_cam_idx]
        target_intrin = self.intrin[target_cam_idx]

        #print(ref_T_a2b, ref_epipole_in2_2dpt, ref_intrin, target_intrin, ref_box)
        epipolar_line = self.calc_epipolar_line(ref_T_a2b, ref_epipole_in2_2dpt, ref_intrin, target_intrin, ref_box)
        return epipolar_line

    def solve_system_of_equations(self, eq1, eq2):
        a1, b1, c1 = eq1
        a2, b2, c2 = eq2

        A = np.array([[a1, b1], [a2, b2]]) 
        b = np.array([-c1, -c2])

        x, y = np.linalg.solve(A, b)
        return x, y

    def calc_epipolar_line(self, T_a2b, epipole1_in2_2dpt, cam1_intrin, cam2_intrin, cam1_box): 
        cam1_box = self.resized_box_to_original_box(cam1_box.reshape(-1, 4))
        x1, y1, x2, y2 = cam1_box.reshape(4, )
        bbox1_2dpt = (x1 + x2) / 2, (y1 + y2) / 2

        # bbox 1 in camera 2
        bbox1_3dpt = np.matmul(np.linalg.inv(cam1_intrin), np.array([*bbox1_2dpt, 1]))
        bbox1_3dpt = np.array([*bbox1_3dpt.tolist(), 1])

        bbox1_in2_3dpt = np.matmul(T_a2b, bbox1_3dpt)[:3]
        bbox1_in2_2dpt = np.matmul(cam2_intrin, bbox1_in2_3dpt)
        bbox1_in2_2dpt = bbox1_in2_2dpt[:2] / bbox1_in2_2dpt[2]

        # find epipolar line
        a, b, c = self.find_line(bbox1_in2_2dpt, epipole1_in2_2dpt)
        import math
        if math.isnan(a) :
            print('bbox1_2dpt', bbox1_2dpt)
            print('bbox1_in2_3dpt', bbox1_in2_3dpt)
            print('bbox1_in2_2dpt', bbox1_in2_2dpt)

        return a, b, c

    def check_cross_pnt_valid(self, cross_pnt):
        pass

    def is_pnt_in_view(self, pnt) :
        x, y = pnt
        if -self.view_x_margin < x < (self.width + self.view_x_margin) and -self.view_y_margin < y < (self.height + self.view_y_margin) :
            return True 
        else :
            return False

    def get_box_idx_on_cross_line(self, line1, line2, boxes) : 
        cross_pnt = self.solve_system_of_equations(line1, line2)
        #if not self.check_cross_pnt_valid(cross_pnt):
        #    return [], [], [-1]
        #print(cross_pnt)
        boxes = self.resized_box_to_original_box(boxes)
        boxes_centers = self.get_boxes_centers(boxes)
        dist = self.find_dist_pnt2pnts(cross_pnt, boxes_centers) / self.diag
        valid_idx = np.where(dist < self.max_dist_epiline_cross_to_box)

        is_valid_inst = True
        if self.is_pnt_in_view(cross_pnt) and valid_idx[0].size == 0 :
            is_valid_inst = False

        if valid_idx[0].size :
            is_valid_box = True
        else :
            is_valid_box = False

        return valid_idx, dist, cross_pnt, is_valid_inst, is_valid_box

    def draw_boxes_with_epiline(self, ref_cam_idx, ref_box, target_cam_idx, epipolar_line, boxes) :
        boxes = self.resized_box_to_original_box(boxes)
        boxes_centers = self.get_boxes_centers(boxes)
        dists = self.find_dist_line2pnts(epipolar_line, boxes_centers)
        dists /= self.diag

        original_ref_box = self.resized_box_to_original_box(ref_box.reshape(-1, 4))[0]

        a, b, c = epipolar_line

        for box, dist in zip(boxes, dists) :
            box_center = ((box[0]+box[2])/2, (box[1]+box[3])/2)
            foot = self.find_foot(a, b, c, box_center)
            print('dist', dist, 'thersh', self.max_dist_epiline_to_box)
            self.draw_result(ref_cam_idx, target_cam_idx, original_ref_box, box, foot, a, b, c)

    def get_box_idx_on_epiline(self, epipolar_line, boxes) :
        boxes = self.resized_box_to_original_box(boxes)
        boxes_centers = self.get_boxes_centers(boxes)
        dist = self.find_dist_line2pnts(epipolar_line, boxes_centers)
        dist /= self.diag
        valid_idx = np.where(dist < self.max_dist_epiline_to_box)
        #print(dist, self.max_dist_epiline_to_box)
        return valid_idx, dist

    def find_dist_line2pnts(self, line, pnts):
        a, b, c = line
        dist = []
        for pnt in pnts : 
            foot = self.find_foot(a, b, c, pnt)
            dist.append(self.find_dist(pnt, foot))
        return np.array(dist)
        
    def find_dist_pnt2pnts(self, pnt, pnts):
        return np.array( [self.find_dist(pnt, cur_pnt) for cur_pnt in pnts] ) 

    def get_boxes_centers(self, bboxes):
        x_center = (bboxes[:, 0] + bboxes[:, 2]) / 2
        y_center = (bboxes[:, 1] + bboxes[:, 3]) / 2
        return np.column_stack([x_center, y_center])

import math
import numpy as np
import random
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utility import iou, union, get_concat_img, CALC_REGR
import cv2

class RPN_GT_CALCULATOR:
    def __init__(self, args):
        self.num_valid_cam = args.num_valid_cam
        self.resized_width, self.resized_height = args.resized_width, args.resized_height
        self.downscale = float(args.rpn_stride) 
        self.anchor_wh = args.anchor_wh
        self.rpn_max_num_sample = args.rpn_max_num_sample
        self.rpn_max_overlap = args.rpn_max_overlap
        self.rpn_min_overlap = args.rpn_min_overlap
        self.rpn_std_scaling = args.rpn_std_scaling
        
        self.output_width, self.output_height = map(lambda a : int(a/self.downscale), [self.resized_width, self.resized_height])
        self.num_anchors = len(self.anchor_wh) # 3x3=9

        self.rpn_num_pos_max = self.rpn_max_num_sample/2
        self.calc_regr = CALC_REGR(self.rpn_std_scaling)

    def get_batch(self, gt_insts_batch):
        rpn_gt_batch = [[] for _ in range(self.num_valid_cam*2)]
        for gt_insts in gt_insts_batch :
            result = self.get_rpn_gt(gt_insts)
            for i in range(self.num_valid_cam*2):
                rpn_gt_batch[i].append(result[i])
        rpn_gt_batch = list(map(lambda a : np.stack(a, 0), rpn_gt_batch))
        return rpn_gt_batch
 
    def sort_by_cam(self, gt_insts):
        boxes_sorted_by_cam = [[] for _ in range(self.num_valid_cam)] 
        for gt_inst in gt_insts:
            for cam_idx, box in list(gt_inst['resized_box'].items()) :
                boxes_sorted_by_cam[cam_idx].append(box)
        return boxes_sorted_by_cam

    def get_rpn_gt(self, gt_insts):
        boxes = self.sort_by_cam(gt_insts)
        result_list = []
        for cam_idx in range(self.num_valid_cam) :
            rpn_gt_cls, rpn_gt_regr = self.calc_rpn_gt(boxes[cam_idx])
            result_list.extend([rpn_gt_cls, rpn_gt_regr])
        return result_list

    def calc_rpn_gt(self, gt_boxes) :
        gt_boxes = np.array(gt_boxes)
        num_gt_boxes = len(gt_boxes)
        #if(num_gt_boxes) : gt_boxes[:,[2,1]] = gt_boxes[:,[1,2]] #x1, y1, x2, y2

        pos_anchor_idx, neg_anchor_idx = np.zeros((0, 3), dtype='int32'), np.zeros((0, 3), dtype='int32')
        pos_anchor_box, pos_gt_box = np.zeros((0, 4)), np.zeros((0, 4))

        best_iou_for_gt_box = np.zeros((num_gt_boxes, )).astype(np.float32)
        best_anchor_box_for_gt_box = np.zeros((num_gt_boxes, 4))
        best_anchor_idx_for_gt_box = np.zeros((num_gt_boxes, 3))
        num_anchors_for_gt_box = np.zeros((num_gt_boxes, ))
        neg_anchor_idx_for_gt_box = np.zeros((num_gt_boxes, ), dtype='int32')
    
        x1_anc_list = []
        y1_anc_list = []
        for ka, (anchor_w, anchor_h) in enumerate(self.anchor_wh):
            for ix in range(self.output_width):                  
                x1_anc = self.downscale * (ix + 0.5) - anchor_w / 2
                x1_anc_list.append(x1_anc)
                x2_anc = x1_anc + anchor_w

                if x1_anc < 0 or x2_anc > self.resized_width:
                    continue
                    
                for jy in range(self.output_height):
                    y1_anc = self.downscale * (jy + 0.5) - anchor_h / 2
                    y1_anc_list.append(y1_anc)
                    y2_anc = y1_anc + anchor_h

                    # ignore boxes that go across image boundaries
                    if y1_anc < 0 or y2_anc > self.resized_height:
                        continue
                    
                    anchor_box = [x1_anc, y1_anc, x2_anc, y2_anc]

                    best_iou = 0.0 
                    best_gt_box_idx = -1
                    for gt_box_idx, gt_box in enumerate(gt_boxes):
                        cur_iou = iou(gt_box, anchor_box)
                        if cur_iou > best_iou : 
                            best_iou = cur_iou
                            best_gt_box_idx = gt_box_idx
                        
                        if cur_iou > best_iou_for_gt_box[gt_box_idx]:
                            best_iou_for_gt_box[gt_box_idx] = cur_iou
                            best_anchor_idx_for_gt_box[gt_box_idx] = [jy, ix, ka]
                            best_anchor_box_for_gt_box[gt_box_idx] = anchor_box

                    if best_iou > self.rpn_max_overlap:
                        pos_anchor_idx = np.append(pos_anchor_idx, [[jy, ix, ka]], 0)
                        pos_anchor_box = np.append(pos_anchor_box, [anchor_box], 0)
                        pos_gt_box = np.append(pos_gt_box, [gt_box], 0)
                        num_anchors_for_gt_box[best_gt_box_idx] += 1

                    elif best_iou < self.rpn_min_overlap:
                        if np.array_equal(best_anchor_idx_for_gt_box[best_gt_box_idx], [jy, ix, ka]) :
                            neg_anchor_idx_for_gt_box[best_gt_box_idx] = len(neg_anchor_idx)
                        neg_anchor_idx = np.append(neg_anchor_idx, [[jy, ix, ka]], 0)

        not_covered_gt_box_idx = np.where(num_anchors_for_gt_box == 0)
        additional_pos_anchor_idx, additional_pos_anchor_box, additional_pos_gt_box, neg_anchor_idx_to_remove = map(lambda a : a[not_covered_gt_box_idx], [best_anchor_idx_for_gt_box, best_anchor_box_for_gt_box, gt_boxes, neg_anchor_idx_for_gt_box])
        
        pos_anchor_idx = np.append(pos_anchor_idx, additional_pos_anchor_idx.reshape(-1, 3), 0)
        pos_anchor_box = np.append(pos_anchor_box, additional_pos_anchor_box.reshape(-1, 4), 0)
        pos_gt_box = np.append(pos_gt_box, additional_pos_gt_box.reshape(-1, 4), 0)
        neg_anchor_idx = np.delete(neg_anchor_idx, neg_anchor_idx_for_gt_box, 0)
                      
        num_pos, num_neg = len(pos_anchor_idx), len(neg_anchor_idx)
        '''
        if(num_pos > self.rpn_num_pos_max) : 
            valid_pos_idx = np.random.choice(num_pos, self.rpn_num_pos_max)
            pos_anchor_idx = pos_anchor_idx[valid_pos_idx]
            pos_anchor_box = pos_anchor_box[valid_pos_idx]
            pos_gt_box = pos_gt_box[valid_pos_idx]
            num_pos = len(pos_anchor_idx)

        if(num_pos + num_neg > self.rpn_max_num_sample) : 
            valid_neg_idx = np.random.choice(num_neg, num_pos)
            neg_anchor_idx = neg_anchor_idx[valid_neg_idx]
        '''

        y_is_sample = np.zeros((self.output_height, self.output_width, self.num_anchors))
        y_is_pos = np.zeros((self.output_height, self.output_width, self.num_anchors))
        y_regr = np.zeros((self.output_height, self.output_width, self.num_anchors, 4))

        pos_anchor_idx, neg_anchor_idx = map(lambda a : tuple(a.astype('int32').T), [pos_anchor_idx, neg_anchor_idx])
        y_is_sample[pos_anchor_idx] = 1
        y_is_sample[neg_anchor_idx] = 1
        y_is_pos[pos_anchor_idx] = 1

        regr = self.calc_regr.calc_t(pos_anchor_box, pos_gt_box)
        y_regr[pos_anchor_idx] = regr 
        y_regr = y_regr.reshape(self.output_height, self.output_width, self.num_anchors*4)

        y_rpn_cls = np.concatenate([y_is_sample, y_is_pos], axis=-1)
        y_rpn_regr = np.concatenate([np.repeat(y_is_pos, 4, axis=-1), y_regr], axis=-1)
        
        return y_rpn_cls, y_rpn_regr

    def draw_rpn_gt(self, imgs_in_batch, rpn_gt_batch) : 
        imgs_in_batch = imgs_in_batch.transpose(1, 0, 2, 3, 4)
        rpn_cls_in_batch = [np.concatenate(rpn_gt_batch[::2])]
        cv2.namedWindow("img", cv2.WINDOW_NORMAL) 
        color = (0, 0, 255)
        anchor_wh = np.array(self.anchor_wh)
        result_imgs_batch = []
        for imgs, rpn_clss in zip(imgs_in_batch, rpn_cls_in_batch):
            result_imgs = []
            for i, (img, rpn_cls) in enumerate(zip(imgs, rpn_clss)) :
                pos_anchor_h_idx, pos_anchor_w_idx, pos_anchor_a_idx = np.where(rpn_cls[:, :, self.num_anchors:]==1)

                if(len(pos_anchor_h_idx)):
                    pos_anchor_wh = anchor_wh[pos_anchor_a_idx]
                    W, H = tuple(pos_anchor_wh.T)
                    X1 = self.downscale * (pos_anchor_w_idx + 0.5) - W / 2
                    X2 = X1 + W
                    Y1 = self.downscale * (pos_anchor_h_idx + 0.5) - H / 2
                    Y2 = Y1 + H
                    
                    X1, X2, Y1, Y2 = map(lambda a : np.rint(a).astype('int'), [X1, X2, Y1, Y2])
                    for x1, x2, y1, y2 in zip(X1, X2, Y1, Y2) :
                        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                result_imgs.append(img)
            imgs_to_show = get_concat_img(result_imgs) if len(result_imgs)>1 else result_imgs[0]
            cv2.imshow('img', imgs_to_show)
            cv2.waitKey()
            result_imgs_batch.append(result_imgs)
        return result_imgs_batch

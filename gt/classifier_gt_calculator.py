import math
import numpy as np
import random
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import utility
import cv2
from gt.calc_regr import CALC_REGR

class CLASSIFIER_GT_CALCULATOR:
    def __init__(self, args):
        self.ideal_num_pos = args.num_rois // 2
        self.num_valid_cam = args.num_valid_cam
        self.classifier_std_scaling = args.classifier_std_scaling
        self.num_cls = args.num_cls
        self.min_overlap = args.classifier_min_overlap
        self.max_overlap = args.classifier_max_overlap

        self.num_cls_fg_bg = self.num_cls + 1
        self.cls_one_hot = np.eye(self.num_cls_fg_bg, dtype='int')
       
        self.rpn_stride = args.rpn_stride

        self.calc_regr = CALC_REGR(self.classifier_std_scaling)

        self.regr_size = self.num_valid_cam*self.num_cls*8

        self.args = args
 
    def get_batch(self, *args):
        X_box_batch, Y_cls_batch, Y_regr_batch, num_neg_batch, num_pos_batch = [], [], [], [], []
        for args_one_batch in zip(*args) :
            X_box, Y_cls, Y_regr, iou, ious_list, num_neg, num_pos = self.calc_classifier_gt(*args_one_batch)
            X_box_batch.append(X_box)
            Y_cls_batch.append(Y_cls)
            Y_regr_batch.append(Y_regr)
            num_neg_batch.append(num_neg)
            num_pos_batch.append(num_pos)
        return np.array(X_box_batch), np.array(Y_cls_batch), np.array(Y_regr_batch), np.array(iou), np.array(ious_list), np.array(num_neg_batch), np.array(num_pos_batch) 
       
    def get_gt_insts_box_cls(self, gt_insts):
        num_inst = len(gt_insts)
        boxes = np.zeros((num_inst, self.num_valid_cam, 4))
        cls = np.zeros((num_inst, ), dtype='int')
        is_valid = np.zeros((num_inst, self.num_valid_cam), dtype='uint8')
        for i, gt_inst in enumerate(gt_insts):
            cls[i] =  gt_inst['cls']
            for cam_idx, box in list(gt_inst['resized_box'].items()) :
                boxes[i, cam_idx] = box
                is_valid[i, cam_idx] = 1
        return boxes, cls, is_valid

    def calc_classifier_gt(self, all_pred_boxes, is_pred_box_valid, gt_insts) :
        #all_pred_boxes, (300, num_valid_cam, 4)
        #is_pred_box_valid (300, num_valid_cam)
        #gt_insts, (N, num_valid_cam, 4)

        all_gt_boxes, gt_cls, is_gt_box_valid = self.get_gt_insts_box_cls(gt_insts)
        all_gt_boxes = np.around(all_gt_boxes/self.rpn_stride)
        #all_gt_boxes = all_gt_boxes/self.rpn_stride
        pos_pred_idx, pos_gt_idx, neg_idx = [], [], []
        pos_iou, pos_iou_list, neg_iou, neg_iou_list = [], [], [], []
        for pred_idx, (pred_boxes, is_pred_valid) in enumerate(zip(all_pred_boxes, is_pred_box_valid)):
            best_iou = 0.0 
            best_iou_list = None
            best_gt_idx = -1
            best_is_neg = False
            for gt_idx, (gt_boxes, is_gt_valid) in enumerate(zip(all_gt_boxes, is_gt_box_valid)):
                cur_iou, iou_list, is_neg = utility.mv_iou(pred_boxes, gt_boxes, is_pred_valid, is_gt_valid)

                if cur_iou > best_iou : 
                    best_iou = cur_iou 
                    best_iou_list = iou_list
                    best_gt_idx = gt_idx
                    best_is_neg = is_neg

            if best_iou > self.min_overlap :
                if best_is_neg or best_iou < self.max_overlap :
                    neg_idx.append(pred_idx)
                    neg_iou.append(best_iou)
                    neg_iou_list.append(best_iou_list)
                else :
                    pos_pred_idx.append(pred_idx)
                    pos_gt_idx.append(best_gt_idx)
                    pos_iou.append(best_iou)
                    pos_iou_list.append(best_iou_list)

        pos_pred_idx, pos_gt_idx = np.array(pos_pred_idx, dtype='int'), np.array(pos_gt_idx, dtype='int')
        pos_iou, pos_iou_list = np.array(pos_iou), np.array(pos_iou_list)

        neg_pred_idx  = np.array(neg_idx, dtype='int')
        neg_iou, neg_iou_list = np.array(neg_iou), np.array(neg_iou_list)

        num_pos = len(pos_pred_idx)
        num_neg = len(neg_pred_idx)

        neg_box = all_pred_boxes[neg_pred_idx]
        neg_cls = np.zeros((num_neg, self.num_cls_fg_bg))
        neg_cls[:, -1] = 1
        Y_regr_neg = np.zeros((num_neg, self.regr_size)) 

        if(num_pos) : 
            is_pos_box_valid = is_pred_box_valid[pos_pred_idx]
            is_cam_valid = (is_pos_box_valid > 0)
            is_cam_invalid = np.logical_not(is_cam_valid)

            pos_box = all_pred_boxes[pos_pred_idx]
            pos_box[is_cam_invalid] = -1
            X_box = np.concatenate([neg_box, pos_box], 0)
            pos_gt_cls = gt_cls[pos_gt_idx]
            pos_cls = self.cls_one_hot[pos_gt_cls]
            Y_cls = np.concatenate((neg_cls, pos_cls), 0)
            pos_regr = np.zeros((num_pos, self.num_valid_cam, self.num_cls, 4))
            is_pos_regr_valid = np.zeros((num_pos,self.num_valid_cam, self.num_cls, 4))
            pos_gt_box = all_gt_boxes[pos_gt_idx]

            valid_pred_box = pos_box[is_cam_valid]
            valid_gt_box = pos_gt_box[is_cam_valid] 

            cls_idx = np.repeat(pos_gt_cls, self.num_valid_cam).reshape(num_pos, self.num_valid_cam)
            cls_idx = cls_idx[is_cam_valid]
            order_idx, cam_idx = np.where(is_cam_valid==1)
            valid_idx = (order_idx, cam_idx, cls_idx)
            pos_regr[valid_idx] = self.calc_regr.calc_t(valid_pred_box, valid_gt_box)
            is_pos_regr_valid[valid_idx] = 1
            Y_regr_pos = np.stack([is_pos_regr_valid, pos_regr], 1)
            Y_regr_pos = Y_regr_pos.reshape((num_pos, -1))
            Y_regr = np.concatenate([Y_regr_neg, Y_regr_pos], 0)

            iou = np.concatenate([neg_iou, pos_iou], 0)
            iou_list = np.concatenate([neg_iou_list, pos_iou_list], 0)

        else :
            X_box = neg_box
            Y_cls = neg_cls
            Y_regr = Y_regr_neg
            iou = neg_iou
            iou_list = neg_iou_list

        X_box[:, :, 2] -= X_box[:, :, 0]
        X_box[:, :, 3] -= X_box[:, :, 1]

        random_X_box, random_Y_cls, random_Y_regr, random_iou, random_iou_list = self.get_random_samples(X_box, Y_cls, Y_regr, iou, iou_list, num_neg, num_pos)

        return random_X_box, random_Y_cls, random_Y_regr, random_iou, random_iou_list, num_neg, num_pos

    def get_random_samples(self, X2, Y1, Y2, iou, iou_list, num_neg, num_pos):
        neg_samples = np.arange(0, num_neg)
        pos_samples = np.arange(num_neg, num_neg+num_pos)
        if num_pos > self.ideal_num_pos : 
            pos_samples= np.random.choice(pos_samples, self.ideal_num_pos, replace=False)
            num_pos = len(pos_samples)
        num_rest = self.args.num_rois - num_pos
        if num_neg > num_rest :
            neg_samples = np.random.choice(neg_samples, num_rest, replace=False)
        elif num_neg :
            neg_samples = np.random.choice(neg_samples, num_rest, replace=True)
        
        sel_samples = pos_samples.tolist() + neg_samples.tolist()
        return X2[sel_samples], Y1[sel_samples], Y2[sel_samples], iou[sel_samples], iou_list[sel_samples]

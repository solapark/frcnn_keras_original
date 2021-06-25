import math
import numpy as np
import random
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utility import iou, union, get_concat_img
import cv2
from gt.calc_regr import CALC_REGR

class CLASSIFIER_GT_CALCULATOR:
    def __init__(self, args):
        self.args = args
        self.ideal_num_pos = args.num_rois // 2
        self.num_cam = args.num_cam
        self.classifier_std_scaling = args.classifier_std_scaling
        self.num_cls = args.num_cls
        self.min_overlap = args.classifier_min_overlap
        self.max_overlap = args.classifier_max_overlap

        self.num_cls_fg_bg = self.num_cls + 1
        self.cls_one_hot = np.eye(self.num_cls_fg_bg, dtype='int')
       
        self.rpn_stride = args.rpn_stride

        self.calc_regr = CALC_REGR(self.classifier_std_scaling)

        self.regr_size = self.num_cam*self.num_cls*8
 
    def get_batch(self, *args):
        X_box_batch, Y_cls_batch, Y_regr_batch, num_neg_batch, num_pos_batch = [], [], [], [], []
        for args_one_batch in zip(*args) :
            X_box, Y_cls, Y_regr, num_neg, num_pos = self.calc_classifier_gt(*args_one_batch)
            X_box_batch.append(X_box)
            Y_cls_batch.append(Y_cls)
            Y_regr_batch.append(Y_regr)
            num_neg_batch.append(num_neg)
            num_pos_batch.append(num_pos)
        return np.array(X_box_batch), np.array(Y_cls_batch), np.array(Y_regr_batch), np.array(num_neg_batch), np.array(num_pos_batch) 
       
    def get_gt_insts_box_cls(self, gt_insts):
        num_inst = len(gt_insts)
        boxes = np.zeros((num_inst, self.num_cam, 4))
        cls = np.zeros((num_inst, ), dtype='int')
        is_valid = np.zeros((num_inst, self.num_cam))
        for i, gt_inst in enumerate(gt_insts):
            cls[i] =  gt_inst['cls']
            for cam_idx, box in list(gt_inst['resized_box'].items()) :
                boxes[i, cam_idx] = box
                is_valid[i, cam_idx] = 1
        return boxes, cls, is_valid

    def calc_classifier_gt(self, all_pred_boxes, is_pred_box_valid, gt_insts) :
        #all_pred_boxes, (300, num_cam, 4)
        #is_pred_box_valid (300, num_cam)
        #gt_insts, (N, num_cam, 4)

        all_gt_boxes, gt_cls, is_gt_box_valid = self.get_gt_insts_box_cls(gt_insts)
        all_gt_boxes = np.around(all_gt_boxes/self.rpn_stride)
        '''
        all_gt_boxes = np.array([[[58., 60., 14., 19],[2.,  4., 12., 18.], [40., 42., 20., 23.]]])
        all_gt_boxes[:, :, [2,1]] = all_gt_boxes[:, :, [1,2]]
        is_gt_box_valid = np.array([[1, 0, 1]])
        all_pred_boxes =  np.array([[[58., 60., 14., 19],[2.,  4., 12., 18.], [40., 42., 20., 23.]]])
        all_pred_boxes[:, :, [2,1]] = all_pred_boxes[:, :, [1,2]]
        is_pred_box_valid = np.array([[1, 0, 1]])
        '''
        pos_pred_idx, pos_gt_idx, neg_idx = [], [], []
        for pred_idx, (pred_boxes, is_pred_valid) in enumerate(zip(all_pred_boxes, is_pred_box_valid)):
            best_iou = 0.0 
            best_iou_list = np.array([-1])
            best_gt_idx = -1
            is_neg = 0
            for gt_idx, (gt_boxes, is_gt_valid) in enumerate(zip(all_gt_boxes, is_gt_box_valid)):
                common_cam_idx = np.where((is_gt_valid == is_pred_valid) & (is_pred_valid == 1))
                if common_cam_idx[0].size == 0: 
                    continue
                valid_pred_boxes = pred_boxes[common_cam_idx]
                valid_gt_box = gt_boxes[common_cam_idx]
                cur_iou_list = np.array([iou(gt_box, pred_box) for gt_box, pred_box in zip(valid_gt_box, valid_pred_boxes)])
                cur_iou = np.mean(cur_iou_list)

                if cur_iou > best_iou : 
                    best_iou = cur_iou 
                    best_iou_list = cur_iou_list 
                    best_gt_idx = gt_idx
                    if not np.array_equal(is_gt_valid, is_pred_valid) : is_neg = 1

            if is_neg :
                    neg_idx.append(pred_idx)

            elif (best_iou_list > self.max_overlap).all() :
                    pos_pred_idx.append(pred_idx)
                    pos_gt_idx.append(best_gt_idx)

            elif (best_iou_list > self.min_overlap).all() :
                    neg_idx.append(pred_idx)

        pos_pred_idx, pos_gt_idx = np.array(pos_pred_idx, dtype='int'), np.array(pos_gt_idx, dtype='int')
        neg_pred_idx  = np.array(neg_idx, dtype='int')
        num_pos = len(pos_pred_idx)
        num_neg = len(neg_pred_idx)

        neg_box = all_pred_boxes[neg_pred_idx]
        neg_cls = np.zeros((num_neg, self.num_cls_fg_bg))
        neg_cls[:, -1] = 1
        Y_regr_neg = np.zeros((num_neg, self.regr_size)) 
        if(num_pos) : 
            is_pos_box_valid = is_pred_box_valid[pos_pred_idx]
            is_cam_valid = (is_pos_box_valid == 1)
            is_cam_invalid = np.logical_not(is_cam_valid)

            pos_box = all_pred_boxes[pos_pred_idx]
            pos_box[is_cam_invalid] = -1
            X_box = np.concatenate([neg_box, pos_box], 0)
            pos_gt_cls = gt_cls[pos_gt_idx]
            pos_cls = self.cls_one_hot[pos_gt_cls]
            Y_cls = np.concatenate((neg_cls, pos_cls), 0)

            #pos_regr = np.zeros((num_pos, self.num_cam, self.num_cls, 4))
            #is_pos_regr_valid = np.zeros((num_pos, self.num_cam, self.num_cls, 4))
            pos_regr = np.zeros((num_pos, self.num_cam, self.num_cls, 4))
            is_pos_regr_valid = np.zeros((num_pos,self.num_cam, self.num_cls, 4))
            pos_gt_box = all_gt_boxes[pos_gt_idx]

            valid_pred_box = pos_box[is_cam_valid]
            valid_gt_box = pos_gt_box[is_cam_valid] 

            cls_idx = np.repeat(pos_gt_cls, self.num_cam).reshape(num_pos, self.num_cam)
            cls_idx = cls_idx[is_cam_valid]
            order_idx, cam_idx = np.where(is_cam_valid==1)
            valid_idx = (order_idx, cam_idx, cls_idx)
            pos_regr[valid_idx] = self.calc_regr.calc_t(valid_pred_box, valid_gt_box)
            is_pos_regr_valid[valid_idx] = 1
            Y_regr_pos = np.stack([is_pos_regr_valid, pos_regr], 1)
            Y_regr_pos = Y_regr_pos.reshape((num_pos, -1))
            Y_regr = np.concatenate([Y_regr_neg, Y_regr_pos], 0)
        else :
            X_box = neg_box
            Y_cls = neg_cls
            Y_regr = Y_regr_neg

        X_box[:, :, 2] -= X_box[:, :, 0]
        X_box[:, :, 3] -= X_box[:, :, 1]

        random_X_box, random_Y_cls, random_Y_regr = self.get_random_samples(X_box, Y_cls, Y_regr, num_neg, num_pos)

        return random_X_box, random_Y_cls, random_Y_regr, num_neg, num_pos

    def get_random_samples(self, X2, Y1, Y2, num_neg, num_pos):
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

        return X2[sel_samples], Y1[sel_samples], Y2[sel_samples]


if __name__ == '__main__' :
    random.seed(1)
    from dataloader import DATALOADER
    from option import args

    mode = 'train'
    dl = DATALOADER(args, mode)
    imgs_in_batch, labels_in_batch = dl[0]

    classifier_gt_calculator = CLASSIFIER_GT_CALCULATOR(args)

    import pickle
    with open('/home/sap/frcnn_keras/mv_train_two_classfier_gt.pickle', 'rb') as f:
        result = pickle.load(f)
    pred_boxes, x_roi_gt, y_cls_gt, y_regr_gt = result

    with open('/home/sap/frcnn_keras/pred_box_is_valid.pickle', 'rb') as f:
        pred_box_is_valid = pickle.load(f)

    pred_box_is_valid = np.ones((300, 3))

    all_pred_boxes = pred_boxes
    is_pred_box_valid = pred_box_is_valid
    gt_insts = labels_in_batch[0]

    x_roi_pred, y_cls_pred, y_regr_pred = classifier_gt_calculator.calc_classifier_gt(all_pred_boxes, is_pred_box_valid, gt_insts)
    print('x_roi_pred.shape', x_roi_pred.shape, 'y_cls_pred.shape', y_cls_pred.shape, 'y_regr_pred.shape', y_regr_pred.shape)

    all_pred_boxes_batch, is_pred_box_valid_batch = list(map(lambda a : np.expand_dims(a, 0), [all_pred_boxes, is_pred_box_valid]))
    x_roi_pred_batch, y_cls_pred_batch, y_regr_pred_batch = classifier_gt_calculator.get_batch(all_pred_boxes_batch, is_pred_box_valid_batch, labels_in_batch)
    print(np.array_equal(x_roi_pred_batch[0], x_roi_pred), np.array_equal(y_cls_pred_batch[0], y_cls_pred), np.array_equal(y_regr_pred_batch[0], y_regr_pred)) 

    print('x_roi_gt.shape', x_roi_gt.shape, 'y_cls_gt.shape', y_cls_gt.shape, 'y_regr_gt.shape', y_regr_gt.shape)

    print('pred x_roi')
    print('x_roi_pred\n', x_roi_pred)
    print('x_roi_gt\n', x_roi_gt)
    if np.array_equal(x_roi_pred, x_roi_gt) : print('true')
    else : print('false\n\n')

    print('pred y_cls')
    print('y_cls_pred\n', y_cls_pred)
    print('y_cls_gt\n', y_cls_gt)
    if np.array_equal(y_cls_pred, y_cls_gt) : print('true')
    else : print('false\n\n')

    print('pred y_regr')
    print('y_regr_pred\n', y_regr_pred)
    print('y_regr_gt\n', y_regr_gt)
    if np.array_equal(y_regr_pred, y_regr_gt) : print('true')
    else : print('false')

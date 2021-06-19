from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam, SGD, RMSprop

from importlib import import_module
import numpy as np
import random

from loss import losses
from gt import roi_helpers
from gt.rpn_gt_calculator import RPN_GT_CALCULATOR
from gt.classifier_gt_calculator import CLASSIFIER_GT_CALCULATOR
from reid.reid import REID
from reid.reid_gt_calculator import REID_GT_CALCULATOR

import utility
import tmp

def make_model(args):
    return MV_FRCNN(args)

class MV_FRCNN:
    def __init__(self, args):
        self.args = args
        self.class_list = list(args.class_mapping.keys())
        self.num_anchors = args.num_anchors
        self.mode = args.mode

        base_net = import_module('model.' + args.base_net.lower()).make_model(args)
        self.model_rpn, self.model_ven, self.model_classifier, self.model_all = self.make_model(args, base_net)
        self.compile()

        self.rpn_gt_calculator = RPN_GT_CALCULATOR(args)
        self.reid = REID(args)
        self.reid_gt_calculator = REID_GT_CALCULATOR(args)
        self.classifier_gt_calculator = CLASSIFIER_GT_CALCULATOR(args)

        
    def save(self, path):
        self.model_all.save_weights(path)

    def load(self, path):
        self.model_rpn.load_weights(path, by_name=True)
        self.model_ven.load_weights(path, by_name=True)
        self.model_classifier.load_weights(path, by_name=True)

    def compile(self):
        if(self.mode == 'train'):
            self.train_compile()
        else :
            self.test_compile()

    def train_compile(self):
        optimizer = Adam(lr=1e-5)
        optimizer_view_invariant = Adam(lr=1e-5)
        optimizer_classifier = Adam(lr=1e-5)
        rpn_loss = []
        for i in range(self.args.num_valid_cam) : 
            rpn_loss.extend([losses.rpn_loss_cls(self.args.num_anchors), losses.rpn_loss_regr(self.args.num_anchors)])
        self.model_rpn.compile(optimizer=optimizer, loss=rpn_loss)
        self.model_ven.compile(optimizer=optimizer_view_invariant, loss=losses.view_invariant_loss(self.args.ven_loss_alpha))
        self.model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(self.args.num_cls_with_bg-1, self.args.num_valid_cam)])
        self.model_all.compile(optimizer='sgd', loss='mae')

    '''
    def test_compile(self):
        self.rpn.compile(optimizer='sgd', loss='mse')
        self.classifier.compile(optimizer='sgd', loss='mse')
    '''

    def make_model(self, args, basenet):
        input_shape_img = (None, None, 3)

        img_input = []
        roi_input = []
        for i in range(args.num_valid_cam) : 
            img_input.append(Input(shape=input_shape_img))
            roi_input.append(Input(shape=(None, 4)))

        shared_layer = basenet.nn_base_model()
        shared_layers = []
        for i in range(args.num_valid_cam):
            shared_layers.append(shared_layer(img_input[i]))
            
        # define the RPN, built on the base layers
        rpn_body, rpn_class, rpn_regr = basenet.rpn_layer_model(args.num_anchors)
        view_invariant_layer = basenet.view_invariant_layer_model(args.grid_rows, args.grid_cols, args.num_anchors, args.view_invar_feature_size)

        rpns = []
        view_invariants = []
        for i in range(args.num_valid_cam) :
            body = rpn_body(shared_layers[i])
            cls = rpn_class(body)
            regr = rpn_regr(body)
            view_invariant = view_invariant_layer(body)
            rpns.extend([cls, regr])
            view_invariants.append(view_invariant)

        view_invariant_conc = basenet.view_invariant_conc_layer(view_invariants)
        classifier = basenet.classifier_layer(shared_layers, roi_input, args.num_rois, args.classifier_num_input_features, args.num_valid_cam, args.num_cls_with_bg)

        model_rpn = Model(img_input, rpns)
        model_ven = Model(img_input, view_invariant_conc)
        classifier_input = img_input + roi_input
        model_classifier = Model(classifier_input, classifier)

        model_all = Model(classifier_input, rpns + [view_invariant_conc] + classifier)

        return model_rpn, model_ven, model_classifier, model_all

    def get_train_model(self):
        return self.model_rpn, self.model_classifier, self.model_all

    '''
    def get_test_model(self):
        return self.model_rpn, self.model_classifier_only
    '''

    def train(self, X, Y, img_data):
        loss_rpn = self.model_rpn.train_on_batch(X, Y)
        P_rpn = self.model_rpn.predict_on_batch(X)

        R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], self.args, K.common.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
        # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
        X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, self.args, args.class_mapping)

        num_pos_samples = 0
        if X2 is not None:
            X2, Y1, Y2, num_pos_samples = roi_helpers.get_classifier_samples(X2, Y1, Y2, self.args.num_rois)
            sample_X2, sample_Y1, sample_Y2 = model_classifier.get_random_sample(X2, Y1, Y2, num_pos_samples)
            loss_class = model_classifier.train_on_batch([X, X2], [Y1, Y2])
            loss = loss_rpn[1:3] + loss_class[1:3] 
            loss.append(sum(loss))
        return loss, num_pos_samples

    def predict(self, X):
        [Y1, Y2, F] = self.rpn.predict(X)
        R = roi_helpers.rpn_to_roi(Y1, Y2, self.args, K.common.image_dim_ordering(), overlap_thresh=0.5)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // self.args.num_rois + 1):
            ROIs = np.expand_dims(R[self.args.num_rois * jk:self.args.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // self.args.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], self.args.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            #[P_cls, P_regr] = self.classifier.predict([F, ROIs])
            [P_cls, P_regr] = self.classifier_only.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):

                if np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = self.class_list[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= self.args.classifier_regr_std[0]
                    ty /= self.args.classifier_regr_std[1]
                    tw /= self.args.classifier_regr_std[2]
                    th /= self.args.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]
                det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}
                all_dets.append(det)

        return all_dets

    def train_batch(self, X, Y, debug_img):
        loss = np.array([np.Inf]*6)
        num_pos_samples = 0

        X_list = list(X)

        rpn_gt_batch = self.rpn_gt_calculator.get_batch(Y)
        loss_rpn = self.model_rpn.train_on_batch(X_list, rpn_gt_batch)
        #self.rpn_gt_calculator.draw_rpn_gt(X, rpn_gt_batch)

        loss[0:2] = loss_rpn[1:3]
        loss[0:2] /= self.args.num_valid_cam

        P_rpn = self.model_rpn.predict_on_batch(list(X))

        R_list = []
        for i in range(self.args.num_valid_cam):
            cam_idx = i*2
            rpn_probs = P_rpn[cam_idx]
            rpn_boxs = P_rpn[cam_idx+1]
            R = tmp.rpn_to_roi(rpn_probs, rpn_boxs, self.args, K.common.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes = self.args.num_nms)
            R_list.append(R)
 

        pred_box_idx = np.array([R[1] for R in R_list]) #(num_cam, 300, 3)
        pred_box = np.array([R[2] for R in R_list])
        pred_box_prob = np.array([R[3] for R in R_list])
        cam_idx_arr = np.repeat(np.arange(self.args.num_valid_cam), self.args.num_nms).reshape(self.args.num_valid_cam, self.args.num_nms, 1)
        pred_box_idx = np.concatenate((cam_idx_arr, pred_box_idx), axis = 2)

        view_invariant_features = self.model_ven.predict_on_batch(list(X))
        all_box_emb = np.squeeze(view_invariant_features)
        pred_box_emb = all_box_emb[tuple(pred_box_idx.T)].transpose((1, 0, 2))

        pred_box_batch, pred_box_idx_batch, all_box_emb_batch, pred_box_emb_batch, pred_box_prob_batch = list(map(lambda a : np.expand_dims(a, 0), [pred_box, pred_box_idx, all_box_emb, pred_box_emb, pred_box_prob]))

        ref_pos_neg_idx_batch = self.reid_gt_calculator.get_batch(pred_box_batch, pred_box_idx_batch, all_box_emb_batch, Y)
        #self.reid_gt_calculator.draw_anchor_pos_neg(R_list, ref_pos_neg_idx_batch, debug_img) 

        if(ref_pos_neg_idx_batch.size == 0):
            return loss, num_pos_samples
 
        ref_pos_neg_idx_batch = np.expand_dims(np.expand_dims(ref_pos_neg_idx_batch, -1), -1)
        vi_loss = self.model_ven.train_on_batch(X_list, ref_pos_neg_idx_batch)
        loss[2] = vi_loss

        reid_box_pred_batch, is_valid_batch = self.reid.get_batch(pred_box_batch, pred_box_emb_batch, pred_box_prob_batch)
        ref_cam_batch = self.reid.get_ref_cam_idx_batch(pred_box_prob_batch, pred_box_batch, pred_box_emb_batch)

        X2, Y1, Y2 = self.classifier_gt_calculator.get_batch(reid_box_pred_batch, is_valid_batch, Y)

        if X2 is not None:
            X2, Y1, Y2, num_pos_samples = roi_helpers.get_classifier_samples(X2, Y1, Y2, self.args.num_rois)
            X2_list = list(X2.transpose(2, 0, 1, 3))
            loss_class = self.model_classifier.train_on_batch(X_list+X2_list, [Y1, Y2])

            loss[3:5] = loss_class[1:3]
            loss[-1] = loss[:-1].sum()

            #cls_box, cls_prob = utility.classfier_output_to_box_prob(np.array(X2_list), Y1, Y2, self.args, 0, self.args.num_valid_cam, False)
            #utility.draw_cls_box_prob(debug_img, cls_box, cls_prob, self.args, self.args.num_valid_cam, is_nms=False)

        return loss, num_pos_samples

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

        self.reid = REID(args)

        if(self.mode == 'train') :
            self.rpn_gt_calculator = RPN_GT_CALCULATOR(args)
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

    def test_compile(self):
        self.model_rpn.compile(optimizer='sgd', loss='mse')
        self.model_ven.compile(optimizer='sgd', loss='mse')
        self.model_classifier.compile(optimizer='sgd', loss='mse')

    def make_model(self, args, basenet):
        if(args.mode == 'train'):
            model_rpn, model_ven, model_classifier, model_all =  self.make_train_model(args, basenet)
        else:
            model_rpn, model_ven, model_classifier =  self.make_test_model(args, basenet)
            model_all = None
        return model_rpn, model_ven, model_classifier, model_all

    def make_train_model(self, args, basenet):
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

    def make_test_model(self, args, basenet):
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, args.shared_layer_channels)
        input_shape_rpn_body = (None, None, args.rpn_body_channels)

        img_input = []
        roi_input = []
        feature_map_input = []
        rpn_body_input = []
        for i in range(args.num_valid_cam) : 
            img_input.append(Input(shape=input_shape_img))
            roi_input.append(Input(shape=(args.num_rois, 4)))
            feature_map_input.append(Input(shape=input_shape_features))
            rpn_body_input.append(Input(shape=input_shape_rpn_body))

        # define the base network (VGG here, can be Resnet50, Inception, etc)
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
            rpns.extend([cls, regr, shared_layers[i], body])
            view_invariant = view_invariant_layer(rpn_body_input[i])
            view_invariants.append(view_invariant)

        view_invariant_conc = basenet.view_invariant_conc_layer(view_invariants)
        classifier = basenet.classifier_layer(feature_map_input, roi_input, args.num_rois, args.shared_layer_channels, args.num_valid_cam, nb_classes=args.num_cls_with_bg)

        model_rpn = Model(img_input, rpns)
        model_ven = Model(rpn_body_input, view_invariant_conc)
        classifier_input = feature_map_input + roi_input
        model_classifier = Model(classifier_input, classifier)

        return model_rpn, model_ven, model_classifier

    def predict_batch(self, X):
        X_list = list(X)

        P_rpn = self.model_rpn.predict_on_batch(X_list)

        F_list = [P_rpn[i*4+2] for i in range(self.args.num_valid_cam)]
        rpn_body_list = [P_rpn[i*4+3] for i in range(self.args.num_valid_cam)]

        R_list = []
        for i in range(self.args.num_valid_cam):
            cam_idx = i*4
            rpn_probs = P_rpn[cam_idx]
            rpn_boxs = P_rpn[cam_idx+1]
            R = tmp.rpn_to_roi(rpn_probs, rpn_boxs, self.args, K.common.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes = self.args.num_nms)
            R_list.append(R)
           

        pred_box_idx = np.array([R[1] for R in R_list]) #(num_cam, 300, 3)
        pred_box = np.array([R[2] for R in R_list])
        pred_box_prob = np.array([R[3] for R in R_list])

        cam_idx_arr = np.repeat(np.arange(self.args.num_valid_cam), self.args.num_nms).reshape(self.args.num_valid_cam, self.args.num_nms, 1)
        pred_box_idx = np.concatenate((cam_idx_arr, pred_box_idx), axis = 2)


        view_invariant_features = self.model_ven.predict_on_batch(rpn_body_list)
        all_box_emb = np.squeeze(view_invariant_features)
        pred_box_emb = all_box_emb[tuple(pred_box_idx.T)].transpose((1, 0, 2))

        pred_box_batch, pred_box_emb_batch, pred_box_prob_batch = list(map(lambda a : np.expand_dims(a, 0), [pred_box, pred_box_emb, pred_box_prob]))

        reid_box_pred_batch, is_valid_batch = self.reid.get_batch(pred_box_batch, pred_box_emb_batch, pred_box_prob_batch) #(B, 300, cam, 4), (B, 300, 3)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        reid_box_pred_batch[:, :, :, 2] -= reid_box_pred_batch[:, :, :, 0]
        reid_box_pred_batch[:, :, :, 3] -= reid_box_pred_batch[:, :, :, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {cls_name : [ [] for _ in range(self.args.num_valid_cam) ] for cls_name in self.args.class_mapping_without_bg.keys()}
        probs = {cls_name : [] for cls_name in self.args.class_mapping_without_bg.keys()}

        num_reid_intsts = reid_box_pred_batch.shape[1]
        for jk in range(num_reid_intsts // self.args.num_rois + 1):
            ROIs_list = []
            ROIs_all_cam = reid_box_pred_batch[:, self.args.num_rois * jk:self.args.num_rois * (jk + 1)]

            if ROIs_all_cam.shape[1] == 0:
                break

            for cam_idx in range(self.args.num_valid_cam):
                ROIs = ROIs_all_cam[:, :, cam_idx]

                if jk == num_reid_intsts // self.args.num_rois:
                    # pad R
                    curr_shape = ROIs.shape
                    target_shape = (curr_shape[0], self.args.num_rois, curr_shape[2])
                    ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                    ROIs_padded[:, :curr_shape[1], :] = ROIs
                    ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                    ROIs = ROIs_padded

                ROIs_list.append(ROIs)

            [P_cls, P_regr] = self.model_classifier.predict(F_list+ROIs_list)

            cur_bboxes, cur_probs = utility.classfier_output_to_box_prob(np.array(ROIs_list), P_cls, P_regr, self.args, 0, self.args.num_valid_cam, False, is_exclude_bg=True)

            for cls_name in cur_bboxes.keys():
                for cam_idx in range(self.args.num_valid_cam):
                    bboxes[cls_name][cam_idx].extend(cur_bboxes[cls_name][cam_idx])
                probs[cls_name].extend(cur_probs[cls_name])

        all_dets = [[] for _ in range(self.args.num_valid_cam)]
        for key in bboxes:
            cur_bboxes = np.array(bboxes[key])
            if not cur_bboxes.size : continue
            cur_probs = np.array(probs[key])
            new_boxes_all_cam, new_probs = utility.non_max_suppression_fast_multi_cam(cur_bboxes, cur_probs, overlap_thresh=0.5)
            for cam_idx in range(self.args.num_valid_cam) : 
                new_boxes = new_boxes_all_cam[cam_idx] #(num_box, 4)
                for jk in range(new_boxes.shape[0]):
                    (x1, y1, x2, y2) = new_boxes[jk,:]
                    if(x1 == -self.args.rpn_stride) :
                        continue 
                    det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}
                    all_dets[cam_idx].append(det)
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

        X2, Y1, Y2, num_neg_samples, num_pos_samples = self.classifier_gt_calculator.get_batch(reid_box_pred_batch, is_valid_batch, Y)
        num_pos_samples = num_pos_samples[0]

        if len(X2):
            #X2, Y1, Y2 = roi_helpers.get_classifier_samples(X2, Y1, Y2, self.args.num_rois, num_neg_samples, num_pos_samples)
            X2_list = list(X2.transpose(2, 0, 1, 3))
            loss_class = self.model_classifier.train_on_batch(X_list+X2_list, [Y1, Y2])

            loss[3:5] = loss_class[1:3]
            loss[-1] = loss[:-1].sum()

            #cls_box, cls_prob = utility.classfier_output_to_box_prob(np.array(X2_list), Y1, Y2, self.args, 0, self.args.num_valid_cam, False)
            #utility.draw_cls_box_prob(debug_img, cls_box, cls_prob, self.args, self.args.num_valid_cam, is_nms=False)

        return loss, num_pos_samples

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
from reid.reid import REID
from reid.reid_gt_calculator import REID_GT_CALCULATOR
from gt.classifier_gt_calculator import CLASSIFIER_GT_CALCULATOR

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

        '''
        if self.args.freeze_rpn :
            for layer in self.model_rpn.layers:
                layer.trainable = False
        '''

        self.reid = REID(args)

        self.rpn_gt_calculator = RPN_GT_CALCULATOR(args)
        self.reid_gt_calculator = REID_GT_CALCULATOR(args)
        self.classifier_gt_calculator = CLASSIFIER_GT_CALCULATOR(args)

        self.cam_idx_arr = np.repeat(np.arange(self.args.num_valid_cam), self.args.num_nms).reshape(self.args.num_valid_cam, self.args.num_nms, 1)
       
    def save(self, path):
        self.model_all.save_weights(path)

    def load(self, path):
        self.model_rpn.load_weights(path, by_name=True)
        self.model_ven.load_weights(path, by_name=True)
        self.model_classifier.load_weights(path, by_name=True)

        '''
        if not self.args.freeze_rpn :
            self.model_rpn.load_weights(path, by_name=True)

        if not self.args.freeze_ven :
            self.model_ven.load_weights(path, by_name=True)

        if not self.args.freeze_classifier :
            self.model_classifier.load_weights(path, by_name=True)
        '''

    def compile(self):
        if self.mode == 'train' :
            if not self.args.freeze_rpn :
                self.rpn_train_compile()

            if not self.args.freeze_ven :
                self.ven_train_compile()

            if not self.args.freeze_classifier :
                self.classifier_train_compile()

            self.model_all.compile(optimizer='sgd', loss='mae')

        elif self.mode == 'val' or self.mode == 'val_models' or self.mode == 'demo' :
            if not self.args.freeze_rpn :
                self.rpn_test_compile()

            if not self.args.freeze_ven :
                self.ven_test_compile()

            if not self.args.freeze_classifier :
                self.classifier_test_compile()

        elif self.mode == 'save_rpn_feature':
            self.rpn_test_compile()

        elif self.mode == 'save_ven_feature':
            if not self.args.freeze_rpn :
                self.rpn_test_compile()
            self.ven_test_compile()

    def rpn_train_compile(self):
        rpn_loss = []
        for i in range(self.args.num_valid_cam) : 
            rpn_loss.extend([losses.rpn_loss_cls(self.args.num_anchors), losses.rpn_loss_regr(self.args.num_anchors)])
        self.model_rpn.compile(optimizer=Adam(lr=1e-5), loss=rpn_loss)

    def rpn_test_compile(self):
        self.model_rpn.compile(optimizer='sgd', loss='mse')

    def ven_train_compile(self) :
        self.model_ven.compile(optimizer=Adam(lr=1e-5), loss=losses.ven_loss(self.args.ven_loss_alpha))

    def ven_test_compile(self) :
        self.model_ven.compile(optimizer='sgd', loss='mse')

    def classifier_train_compile(self) :
        self.model_classifier.compile(optimizer=Adam(lr=1e-5), loss=[losses.class_loss_cls, losses.class_loss_regr(self.args.num_cls_with_bg-1, self.args.num_valid_cam)])

    def classifier_test_compile(self) :
        self.model_classifier.compile(optimizer='sgd', loss='mse')

    def make_model(self, args, basenet):
        model_rpn, model_ven, model_classifier, model_all =  self.make_train_model(args, basenet)

        if args.mode != 'train' or args.freeze_rpn :
            model_rpn, model_ven, model_classifier, model_all =  self.make_test_model(args, basenet)
        else :
            model_rpn, model_ven, model_classifier, model_all =  self.make_train_model(args, basenet)

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
            #view_invariant = view_invariant_layer(body)
            view_invariant = view_invariant_layer(shared_layers[i])
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
        for i in range(args.num_valid_cam) :
            body = rpn_body(shared_layers[i])
            cls = rpn_class(body)
            regr = rpn_regr(body)
            rpns.extend([cls, regr, shared_layers[i], body])
            #view_invariant = view_invariant_layer(rpn_body_input[i])
            #view_invariant = view_invariant_layer(shared_layers[i])
            #view_invariants.append(view_invariant)

        view_invariants = [view_invariant_layer(feature_map_input[i]) for i in range(args.num_valid_cam)]
        view_invariant_conc = basenet.view_invariant_conc_layer(view_invariants)

        classifier = basenet.classifier_layer(feature_map_input, roi_input, args.num_rois, args.shared_layer_channels, args.num_valid_cam, nb_classes=args.num_cls_with_bg)

        model_rpn = Model(img_input, rpns)
        #model_ven = Model(rpn_body_input, view_invariant_conc)
        model_ven = Model(feature_map_input, view_invariant_conc) 
        classifier_input = feature_map_input + roi_input
        model_classifier = Model(classifier_input, classifier) 

        model_all = Model(img_input+classifier_input, rpns + [view_invariant_conc] + classifier)

        return model_rpn, model_ven, model_classifier, model_all

    def parse_rois(self, rois):
        pred_box_idx = np.array([R[1] for R in rois]) #(num_cam, 300, 3)
        pred_box_idx = np.concatenate((self.cam_idx_arr, pred_box_idx), axis = 2)
        pred_box = np.array([R[2] for R in rois])
        pred_box_prob = np.array([R[3] for R in rois])
        return pred_box_idx, pred_box, pred_box_prob

    def to_batch(self, data):
        return np.expand_dims(data, 0)

    def extract_rpn_feature(self, X, X_raw, rpn_result):
        if self.args.freeze_rpn :
            pred_box_idx_batch, pred_box_batch, pred_box_prob_batch, shared_feats = rpn_result[0]

        else :
            pred_box_idx_batch, pred_box_batch, pred_box_prob_batch, shared_feats = self.rpn_predict_batch(list(X), X_raw)      

        return pred_box_idx_batch, pred_box_batch, pred_box_prob_batch, shared_feats

    def ven_predict_batch(self, X ,debug_images, extrins, rpn_result):
        pred_box_idx_batch, pred_box_batch, pred_box_prob_batch, shared_feats = self.extract_rpn_feature(X, debug_images, rpn_result) 

        reid_box_pred_batch, is_valid_batch, _ = self.extract_ven_feature(shared_feats, debug_images, extrins, pred_box_idx_batch, pred_box_batch, pred_box_prob_batch)

        return reid_box_pred_batch, is_valid_batch

    def rpn_predict_batch(self, X, debug_images=None):
        offset = 2 if self.args.mode == 'train' else 4

        P_rpn = self.model_rpn.predict_on_batch(list(X))

        rois = [tmp.rpn_to_roi(P_rpn[i*offset], P_rpn[i*offset+1], self.args, K.common.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes = self.args.num_nms) for i in range(self.args.num_valid_cam)]

        pred_box_idx, pred_box, pred_box_prob = self.parse_rois(rois)

        pred_box_idx_batch = self.to_batch(pred_box_idx)
        pred_box_batch = self.to_batch(pred_box)
        pred_box_prob_batch = self.to_batch(pred_box_prob)

        #nms_list = [R[2] for R in rois]
        #utility.draw_nms(nms_list, debug_img, self.args.rpn_stride) 

        shared_feats = None if self.args.mode == 'train' else [P_rpn[i*offset+2] for i in range(self.args.num_valid_cam)]

        return pred_box_idx_batch, pred_box_batch, pred_box_prob_batch, shared_feats

    def extract_ven_feature(self, inputs, debug_images, extrins, pred_box_idx_batch, pred_box_batch, pred_box_prob_batch):
        view_emb = self.model_ven.predict_on_batch(inputs)
        all_box_emb_batch, pred_box_emb_batch = self.reid_gt_calculator.get_emb_batch(pred_box_batch, pred_box_idx_batch, view_emb) 

        reid_box_pred_batch, is_valid_batch = self.reid.get_batch(pred_box_batch, pred_box_emb_batch, pred_box_prob_batch, extrins, np.array(debug_images).transpose(1, 0, 2, 3, 4))

        return reid_box_pred_batch, is_valid_batch, all_box_emb_batch

    def classifier_predict_batch(self, F_list, reid_box_pred_batch, is_valid_batch, debug_img):
        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        reid_box_pred_batch[:, :, :, 2] -= reid_box_pred_batch[:, :, :, 0]
        reid_box_pred_batch[:, :, :, 3] -= reid_box_pred_batch[:, :, :, 1]

        bboxes = {cls_name : [ [] for _ in range(self.args.num_valid_cam) ] for cls_name in self.args.class_mapping_without_bg.keys()}
        is_valids = {cls_name : [ [] for _ in range(self.args.num_valid_cam) ] for cls_name in self.args.class_mapping_without_bg.keys()}
        probs = {cls_name : [] for cls_name in self.args.class_mapping_without_bg.keys()}

        iou_list = np.zeros((self.args.batch_size, self.args.num_rois, self.args.num_valid_cam))

        num_reid_intsts = reid_box_pred_batch.shape[1]
        for jk in range(num_reid_intsts // self.args.num_rois + 1):
            ROIs_list = []
            ROIs_all_cam = reid_box_pred_batch[:, self.args.num_rois * jk:self.args.num_rois * (jk + 1)]
            is_valids_all_cam = is_valid_batch[:, self.args.num_rois * jk:self.args.num_rois * (jk + 1)]

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

            cur_bboxes, cur_probs, cur_is_valids, _ = utility.classifier_output_to_box_prob(np.array(ROIs_list), P_cls, P_regr, iou_list, self.args, 0, self.args.num_valid_cam, False, is_exclude_bg=True)

            for cls_name in cur_bboxes.keys():
                for cam_idx in range(self.args.num_valid_cam):
                    bboxes[cls_name][cam_idx].extend(cur_bboxes[cls_name][cam_idx])
                    is_valids[cls_name][cam_idx].extend(cur_is_valids[cls_name][cam_idx])
                    
                probs[cls_name].extend(cur_probs[cls_name])

        all_dets = [[] for _ in range(self.args.num_valid_cam)]
        inst_idx = 1
        for key in bboxes:
            cur_bboxes = np.array(bboxes[key])
            if not cur_bboxes.size : continue
            cur_probs = np.array(probs[key])
            cur_is_valids = np.array(is_valids[key])
            new_boxes_all_cam, new_probs, new_is_valids_all_cam = utility.non_max_suppression_fast_multi_cam(cur_bboxes, cur_probs, cur_is_valids, overlap_thresh=0.5)
            for jk in range(new_boxes_all_cam.shape[1]):
                for cam_idx in range(self.args.num_valid_cam) : 
                    (x1, y1, x2, y2) = new_boxes_all_cam[cam_idx, jk]
                    is_valid = new_is_valids_all_cam[cam_idx, jk]
                    if not is_valid : 
                        continue
                    #if(x1 == -self.args.rpn_stride) :
                    #    continue 
                    det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk], 'inst_idx': inst_idx}
                    all_dets[cam_idx].append(det)
                inst_idx += 1
        return all_dets



    def predict_batch(self, X, debug_img, extrins, rpn_result, ven_result):
        if self.args.freeze_rpn :
            pred_box_idx_batch, pred_box_batch, pred_box_prob_batch, shared_feats = rpn_result[0]
        else :
            pred_box_idx_batch, pred_box_batch, pred_box_prob_batch, shared_feats = self.rpn_predict_batch(X, debug_img)

        if self.args.freeze_ven :
            reid_box_pred_batch, is_valid_batch = ven_result[0]
        else :
            reid_box_pred_batch, is_valid_batch, all_box_emb_batch = self.extract_ven_feature(shared_feats, debug_img, extrins, pred_box_idx_batch, pred_box_batch, pred_box_prob_batch)

        all_dets = self.classifier_predict_batch(shared_feats, np.copy(reid_box_pred_batch), is_valid_batch, debug_img)

        #utility.draw_reid(reid_box_pred_batch, is_valid_batch, debug_img, self.args.rpn_stride)

        #result_saver = utility.Result_saver(self.args)
        #image_paths = [{1: '/data1/sap/MessyTable/images/20190921-00001-01-01.jpg', 2: '/data1/sap/MessyTable/images/20190921-00001-01-02.jpg', 3: '/data1/sap/MessyTable/images/20190921-00001-01-03.jpg'}]
        #result_saver.save(debug_img, image_paths, all_dets)

        return all_dets

        """
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

        pred_box_idx = np.concatenate((self.cam_idx_arr, pred_box_idx), axis = 2)


        view_invariant_features = self.model_ven.predict_on_batch(rpn_body_list)
        all_box_emb = np.squeeze(view_invariant_features)
        pred_box_emb = all_box_emb[tuple(pred_box_idx.T)].transpose((1, 0, 2))

        pred_box_batch, pred_box_emb_batch, pred_box_prob_batch = list(map(lambda a : np.expand_dims(a, 0), [pred_box, pred_box_emb, pred_box_prob]))
        """

        """
        import pickle
        data_path = 'data.pickle'
        '''
        target_list = [pred_box_batch, pred_box_emb_batch, pred_box_prob_batch, extrins]
        with open(data_path, 'wb') as f:
            pickle.dump(target_list, f)
        '''
        with open(data_path, 'rb') as f:
            pred_box_batch, pred_box_emb_batch, pred_box_prob_batch, extrins = pickle.load(f)
        """

        '''
        #pred_box_idx_batch, pred_box_batch, pred_box_prob_batch, shared_feats = self.rpn_predict_batch(X, debug_images)

        reid_box_pred_batch, is_valid_batch = self.reid.get_batch(pred_box_batch, pred_box_emb_batch, pred_box_prob_batch, extrins, np.array(debug_images).transpose(1, 0, 2, 3, 4)) #(B, 300, cam, 4), (B, 300, 3)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        reid_box_pred_batch[:, :, :, 2] -= reid_box_pred_batch[:, :, :, 0]
        reid_box_pred_batch[:, :, :, 3] -= reid_box_pred_batch[:, :, :, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {cls_name : [ [] for _ in range(self.args.num_valid_cam) ] for cls_name in self.args.class_mapping_without_bg.keys()}
        is_valids = {cls_name : [ [] for _ in range(self.args.num_valid_cam) ] for cls_name in self.args.class_mapping_without_bg.keys()}
        probs = {cls_name : [] for cls_name in self.args.class_mapping_without_bg.keys()}

        num_reid_intsts = reid_box_pred_batch.shape[1]
        for jk in range(num_reid_intsts // self.args.num_rois + 1):
            ROIs_list = []
            ROIs_all_cam = reid_box_pred_batch[:, self.args.num_rois * jk:self.args.num_rois * (jk + 1)]
            is_valids_all_cam = is_valid_batch[:, self.args.num_rois * jk:self.args.num_rois * (jk + 1)]

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

            cur_bboxes, cur_probs, cur_is_valids = utility.classfier_output_to_box_prob(np.array(ROIs_list), P_cls, P_regr, is_valids_all_cam, self.args, 0, self.args.num_valid_cam, False, is_exclude_bg=True)

            for cls_name in cur_bboxes.keys():
                for cam_idx in range(self.args.num_valid_cam):
                    bboxes[cls_name][cam_idx].extend(cur_bboxes[cls_name][cam_idx])
                    is_valids[cls_name][cam_idx].extend(cur_is_valids[cls_name][cam_idx])
                    
                probs[cls_name].extend(cur_probs[cls_name])

        all_dets = [[] for _ in range(self.args.num_valid_cam)]
        inst_idx = 1
        for key in bboxes:
            cur_bboxes = np.array(bboxes[key])
            if not cur_bboxes.size : continue
            cur_probs = np.array(probs[key])
            cur_is_valids = np.array(is_valids[key])
            new_boxes_all_cam, new_probs, new_is_valids_all_cam = utility.non_max_suppression_fast_multi_cam(cur_bboxes, cur_probs, cur_is_valids, overlap_thresh=0.5)
            for jk in range(new_boxes_all_cam.shape[1]):
                for cam_idx in range(self.args.num_valid_cam) : 
                    (x1, y1, x2, y2) = new_boxes_all_cam[cam_idx, jk]
                    is_valid = new_is_valids_all_cam[cam_idx, jk]
                    if not is_valid : 
                        continue
                    #if(x1 == -self.args.rpn_stride) :
                    #    continue 
                    det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk], 'inst_idx': inst_idx}
                    all_dets[cam_idx].append(det)
                inst_idx += 1
        return all_dets
        '''

    def ven_train_batch(self, inputs, pred_box_batch, pred_box_idx_batch, all_box_emb_batch, Y, debug_img):

        ref_pos_neg_idx_batch = self.reid_gt_calculator.get_batch(pred_box_batch, pred_box_idx_batch, all_box_emb_batch, Y)

        if(ref_pos_neg_idx_batch.size == 0):
            return 0

        ref_pos_neg_idx_batch = np.expand_dims(np.expand_dims(ref_pos_neg_idx_batch, -1), -1)

        #self.reid_gt_calculator.draw_anchor_pos_neg(R_list, ref_pos_neg_idx_batch, debug_img) 
        vi_loss = self.model_ven.train_on_batch(inputs, ref_pos_neg_idx_batch)
        
        return vi_loss

    def classifier_train_batch(self, inputs, reid_box_pred_batch, is_valid_batch, Y, debug_img):
        X2, Y1, Y2, iou, iou_list, num_neg_samples, num_pos_samples = self.classifier_gt_calculator.get_batch(reid_box_pred_batch, is_valid_batch, Y)
        loss = np.zeros((2, ))

        if X2.shape[1] == self.args.num_rois:
            X2_list = list(X2.transpose(2, 0, 1, 3))
            loss = self.model_classifier.train_on_batch(inputs+X2_list, [Y1, Y2])
            loss = loss[1:3]
            if np.isnan(loss[0]):
                print('loss is nan')
            #cls_box, cls_prob, is_valids, cls_iou = utility.classifier_output_to_box_prob(np.array(X2_list), Y1, Y2, iou_list, self.args, 0, self.args.num_valid_cam, False)
            #utility.draw_cls_box_prob(np.array(debug_img).transpose(1, 0, 2, 3, 4)[0], cls_box, cls_prob, cls_iou, self.args, self.args.num_valid_cam, is_nms=False)

        print('pos', num_pos_samples[0], 'neg', num_neg_samples[0])

        return loss, num_pos_samples[0]

    def rpn_train_batch(self, X, debug_img, Y):
        rpn_gt_batch = self.rpn_gt_calculator.get_batch(Y)
        loss_rpn = self.model_rpn.train_on_batch(X_list, rpn_gt_batch)
        #self.rpn_gt_calculator.draw_rpn_gt(np.array(debug_img), rpn_gt_batch)

        loss = loss_rpn / self.args.num_valid_cam
        return loss

    def train_batch(self, X, debug_img, Y, image_paths, extrins, rpn_result, ven_result):
        loss = np.zeros((6, ), dtype=float)
        num_pos_samples = 0

        if self.args.freeze_rpn :
            pred_box_idx_batch, pred_box_batch, pred_box_prob_batch, shared_feats = rpn_result[0]
            inputs = shared_feats 
        else :
            loss[:2] = self.rpn_train_batch(X, Y)
            pred_box_idx_batch, pred_box_batch, pred_box_prob_batch, _ = self.rpn_predict_batch(X, debug_img)
            inputs = list(X)

        if self.args.freeze_ven :
            reid_box_pred_batch, is_valid_batch = ven_result[0]
        else :
            reid_box_pred_batch, is_valid_batch, all_box_emb_batch = self.extract_ven_feature(inputs, debug_img, extrins, pred_box_idx_batch, pred_box_batch, pred_box_prob_batch)
            loss[2] = self.ven_train_batch(inputs, pred_box_batch, pred_box_idx_batch, all_box_emb_batch, Y, debug_img)

        #utility.draw_reid(reid_box_pred_batch, is_valid_batch, debug_img, self.args.rpn_stride)
        if not self.args.freeze_classifier :
            loss[3:5], num_pos_samples = self.classifier_train_batch(inputs, reid_box_pred_batch, is_valid_batch, Y, debug_img)

        loss[-1] = loss[:-1].sum()

        return loss, num_pos_samples

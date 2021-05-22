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
from gt.reid import REID
from gt.reid_gt_calculator import REID_GT_CALCULATOR
from gt.classifier_gt_calculator import CLASSIFIER_GT_CALCULATOR


import utility

def make_model(args):
    return MV_FRCNN(args)

class MV_FRCNN:
    def __init__(self, args):
        self.args = args
        self.class_list = list(args.class_mapping.keys())
        self.num_anchors = args.num_anchors
        self.mode = args.mode

        base_net = import_module('model.' + args.base_net.lower()).make_model(args)
        if args.mode == 'train' :
            self.rpn, self.ven, self.classifier, self.model_all = self.make_train_model(args, base_net)
            self.train_compile() 

            self.rpn_gt_calculator = RPN_GT_CALCULATOR(args)
            self.reid = REID(args)
            self.reid_gt_calculator = REID_GT_CALCULATOR(args)
            self.classifier_gt_calculator = CLASSIFIER_GT_CALCULATOR(args)
 
        else :
            self.rpn, self.ven, self.classifier, self.model_all = self.make_test_model(args, base_net)
            self.test_compile() 

    def get_weight_path(self):
        return self.base_net.get_weight_path()

    def save(self, path):
        self.model_all.save_weights(path)

    def load(self, path):
        self.rpn.load_weights(path, by_name=True)
        self.rpn.layers[-1].load_weights(path, by_name=True)
        self.ven.load_weights(path, by_name=True)
        self.classifier.load_weights(path, by_name=True)

    def train_compile(self):
        optimizer = Adam(lr=1e-5)
        self.rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(self.num_anchors), losses.rpn_loss_regr(self.num_anchors)]*self.args.num_valid_cam)
        self.ven.compile(optimizer=optimizer, loss=losses.ven_loss(self.args.ven_loss_alpha))
        self.classifier.compile(optimizer=optimizer, loss=[losses.class_loss_cls, losses.class_loss_regr(self.args.num_cls, self.args.num_valid_cam)])
        self.model_all.compile(optimizer='sgd', loss='mae')

    def test_compile(self):
        self.rpn.compile(optimizer='sgd', loss='mse')
        self.ven.compile(optimizer='sgd', loss='mse')
        self.classifier.compile(optimizer='sgd', loss='mse')

    def make_train_model(self, args, base_net):
        img_input = Input(shape=(None, None, 3))
        rpn_top_idx= Input(shape = (None, 2), dtype='int32') 
        shared_layer = base_net.nn_base(img_input, trainable=True)
        rpn_cls, rpn_regr, _ = base_net.rpn(shared_layer, self.num_anchors)
        ven_out = base_net.ven(shared_layer, rpn_top_idx, args.ven_feat_size)

        shared_model = Model(img_input, shared_layer)
        rpn_model = Model(img_input, [rpn_cls, rpn_regr])
        ven_model = Model([img_input, rpn_top_idx], ven_out)

        img_inputs = []
        rpn_top_idxs = []
        shared_feats = []
        rpn_outs, ven_outs = [], []
        for i in range(args.num_valid_cam) :
            img_input = Input(shape=(None, None, 3))
            rpn_top_idx = Input(shape = (None, 2), dtype='int32') 
            shared_feat = shared_model(img_input)
            rpn_cls, rpn_regr = rpn_model(img_input)
            ven_out = ven_model([img_input, rpn_top_idx])
            img_inputs.append(img_input)
            rpn_top_idxs.append(rpn_top_idx)
            shared_feats.append(shared_feat)
            rpn_outs.extend([rpn_cls, rpn_regr])
            ven_outs.append(ven_out)

        ven_outs = base_net.ven_conc(ven_outs)
        
        roi_inputs = [Input(shape=(None, 4))  for i in range(args.num_valid_cam)]
        classifier_cls, classifier_regr = base_net.classifier(shared_feats, roi_inputs, args.num_rois, args.num_valid_cam, nb_classes=len(args.class_mapping), trainable=True)

        model_rpn = Model(img_inputs, rpn_outs)
        model_ven = Model(img_inputs + rpn_top_idxs, ven_outs)
        model_classifier = Model(img_inputs + roi_inputs, [classifier_cls, classifier_regr])
        model_all = Model(img_inputs + rpn_top_idxs + roi_inputs, rpn_outs + [ven_outs, classifier_cls, classifier_regr])
        return model_rpn, model_ven, model_classifier, model_all

    def make_test_model(self, base_net):
        feature_map_input = Input(shape=(None, None, 1024))
        classifier_only = base_net.classifier(feature_map_input, roi_input, args.num_rois, args.num_valid_cam, nb_classes=len(args.class_mapping), trainable=True)
        model_classifier_only = Model([feature_map_input, roi_input], classifier_only)
        model_classifier_only.compile(optimizer='sgd', loss='mae')

        if self.args.test_only:
            model_rpn.compile(optimizer='sgd', loss='mae')

        else :
            classifier = base_net.classifier(shared_layers, roi_input, args.num_rois, args.num_valid_cam, nb_classes=len(args.class_mapping), trainable=True)
            model_classifier = Model([img_input, roi_input], classifier)
            model_all = Model([img_input, roi_input], rpn + classifier)

            optimizer = Adam(lr=1e-5)
            model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(self.num_anchors), losses.rpn_loss_regr(self.num_anchors), losses.loss_dummy])
            model_classifier.compile(optimizer=optimizer, loss=[losses.class_loss_cls, losses.class_loss_regr(len(self.args.class_list)-1)], metrics={f'dense_class_{len(self.args.class_list)}': 'accuracy'})

            model_all.compile(optimizer='sgd', loss='mae')
        return model_rpn, model_classifier, model_classifier_only, model_all

    def get_train_model(self):
        return self.model_rpn, self.model_classifier, self.model_all

    def get_test_model(self):
        return self.model_rpn, self.model_classifier_only

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

    def train_batch(self, X, extrin, Y, debug_img):
        X = list(X)
        loss = np.array([np.nan]*6)
        num_pos_samples = 0

        rpn_gt_batch = self.rpn_gt_calculator.get_batch(Y)
        '''
        rpn_gt_cls_batch = np.stack([rpn_gt_batch[2*i] for i in range(self.args.num_valid_cam)]).transpose(1, 0, 2, 3,4)
        self.rpn_gt_calculator.draw_rpn_gt(np.array(debug_img).transpose(1, 0, 2, 3, 4), rpn_gt_cls_batch)
        '''
        loss_rpn = self.rpn.train_on_batch(X, rpn_gt_batch)

        loss_rpn_cls = sum(loss_rpn[1::2])/self.args.num_valid_cam
        loss_rpn_regr = sum(loss_rpn[2::2])/self.args.num_valid_cam
        loss[0:2] = [loss_rpn_cls, loss_rpn_regr]

        '''
        #print(loss_rpn)
        rpn_gt_cls_batch = np.stack([rpn_gt_batch[2*i] for i in range(args.num_valid_cam)]).transpose(1, 0, 2, 3,4)
        rpn_gt_calculator.draw_rpn_gt(np.array(debug_img).transpose(1, 0, 2, 3, 4), rpn_gt_cls_batch)
        '''

        P_rpn = self.rpn.predict_on_batch(X)

        # R_list : list of R, (len=num_valid_cam)
        # R: (_, nms_idx, nms_bboxes, nms_probs) (shape=(H, W, A, 4), shape=(300, 3), shape=(300,4), shape=(300,) )
        R_list = []
        for cam_idx in range(0, self.args.num_valid_cam*2, 2):
            rpn_probs = P_rpn[cam_idx]
            rpn_boxs = P_rpn[cam_idx+1]
            R = roi_helpers.rpn_to_roi(rpn_probs, rpn_boxs, self.args, K.image_data_format(), use_regr=True, overlap_thresh=0.7, max_boxes = self.args.num_nms)
            R[1] = R[1][:, 0:2] #(300, 2)
            R_list.append(R)
                   
        '''
        nms_list = [R[2] for R in R_list]
        debug_img_np = np.array(debug_img).transpose(1, 0, 2, 3, 4)[0]
        utility.draw_nms(nms_list, debug_img_np, self.args.rpn_stride)
        '''

        pred_box_idx = np.array([R[1] for R in R_list]) #(num_valid_cam, 300, 2)
        pred_box = np.array([R[2] for R in R_list]) #(num_valid_cam, 300, 4)
        pred_box_prob = np.array([R[3] for R in R_list]) #(num_valid_cam, 300)
        
        pred_box_idx_batch = list(np.expand_dims(pred_box_idx, 1)) #(num_valid_cam, B, 300, 2)
        pred_box_batch = np.expand_dims(pred_box, 0) #(B, num_valid_cam, 300, 4)
        pred_box_prob_batch = np.expand_dims(pred_box_prob, 0) #(B, num_valid_cam, 300)
        
        view_emb = self.ven.predict_on_batch(X + pred_box_idx_batch) #(B, num_valid_cam, 300, 128)
        ref_pos_neg_idx_batch = self.reid_gt_calculator.get_batch(pred_box_batch, view_emb, Y) #(B, 3, sample, 2)

        if(ref_pos_neg_idx_batch.size == 0):
            return loss, num_pos_samples

        '''
        debug_img_np = np.array(debug_img).transpose(1, 0, 2, 3, 4)[0]
        utility.draw_ref_pos_neg(pred_box, ref_pos_neg_idx_batch[0], debug_img_np, self.args.rpn_stride)
        '''

        ven_loss = self.ven.train_on_batch(X + pred_box_idx_batch, ref_pos_neg_idx_batch)
        loss[2] = ven_loss

        debug_img_np = np.array(debug_img).transpose(1, 0, 2, 3, 4)
        reid_box_pred_batch, is_valid_batch, dist_batch = self.reid.get_batch(pred_box_batch, view_emb, pred_box_prob_batch, extrin, debug_img_np)

        #ref_cam_batch = self.reid.get_ref_cam_idx_batch(pred_box_prob_batch, pred_box_batch, view_emb)
        #debug_img_np = np.array(debug_img).transpose(1, 0, 2, 3, 4)
        #self.reid.draw_reid_batch(reid_box_pred_batch, is_valid_batch, ref_cam_batch, debug_img_np, dist_batch, waitKey=0)

        X2, Y1, Y2, iou_list = self.classifier_gt_calculator.get_batch(reid_box_pred_batch, is_valid_batch, Y)

        '''
        debug_img_np = np.array(debug_img).transpose(1, 0, 2, 3, 4)[0]
        cls_box, cls_prob, cls_iou = utility.classifier_output_to_box_prob(X2.transpose(2, 0, 1, 3), Y1, Y2, iou_list[0], self.args, 0, self.args.num_valid_cam, False)
        utility.draw_cls_box_prob(debug_img_np, cls_box, cls_prob, cls_iou, self.args, num_cam = self.args.num_valid_cam, is_nms=False)
        '''

        if X2 is not None:
            X2, Y1, Y2, num_pos_samples = roi_helpers.get_classifier_samples(X2, Y1, Y2, self.args.num_rois)
            X2 = X2.transpose(2, 0, 1, 3) #(num_valid_cam, B, 4, 4)
            X2 = list(X2)

            '''
            #debug
            cls_box, cls_prob = utility.classfier_output_to_box_prob(X2, Y1, Y2, self.args, 0, 1, False) 
            utility.draw_cls_box_prob(debug_img, cls_box, cls_prob, self.args, 1, is_nms=False)
            '''

            _, loss_class_cls, loss_class_regr = self.classifier.train_on_batch(X + X2, [Y1, Y2])
            loss[3:5] = [loss_class_cls, loss_class_regr]
            loss[5] = loss[:-1].sum()
        '''
        else :
            nms_list = [R]
            utility.draw_nms(nms_list, debug_img, self.args.rpn_stride) 
        '''
        return loss, num_pos_samples

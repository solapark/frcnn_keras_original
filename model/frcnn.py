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
import utility

def make_model(args):
    return FRCNN(args)

class FRCNN:
    def __init__(self, args):
        self.args = args
        self.class_list = list(args.class_mapping.keys())
        self.num_anchors = args.num_anchors
        self.mode = args.mode

        base_net = import_module('model.' + args.base_net.lower()).make_model(args)
        self.rpn, self.classifier, self.classifier_only, self.model_all = self.make_model(args, base_net)

    def get_weight_path(self):
        return self.base_net.get_weight_path()

    def save(self, path):
        self.model_all.save_weights(path)

    def load(self, path):
        self.rpn.load_weights(path, by_name=True)
        self.classifier.load_weights(path, by_name=True)

    def compile(self):
        if(self.mode == 'train'):
            self.train_compile()
        else :
            self.test_compile()

    def train_compile(self):
        optimizer = Adam(lr=1e-5)
        optimizer_classifier = Adam(lr=1e-5)
        self.rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(self.num_anchors), losses.rpn_loss_regr(self.num_anchors)])
        self.classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(self.args.class_list)-1)], metrics={f'dense_class_{len(self.args.class_list)}': 'accuracy'})
        self.model_all.compile(optimizer='sgd', loss='mae')

    def test_compile(self):
        self.rpn.compile(optimizer='sgd', loss='mse')
        self.classifier.compile(optimizer='sgd', loss='mse')

    '''
    def make_model(self, args, base_net):
        roi_input = Input(shape=(None, 4))
        img_input = Input(shape=(None, None, 3))
        shared_layers = base_net.nn_base(img_input, trainable=True)

        rpn = base_net.rpn(shared_layers, self.num_anchors)

        if(args.mode == 'train'):
            rpn = rpn[:2]
            classifier_input = shared_layers
            classifier_model_input = [img_input, roi_input]
        else :
            feature_map_input = Input(shape=(None, None, 1024))
            classifier_input = feature_map_input
            classifier_model_input = [classifier_input, roi_input]

        classifier = base_net.classifier(classifier_input, roi_input, args.num_rois, nb_classes=len(args.class_mapping), trainable=True)
        self.model_rpn = Model(img_input, rpn)
        self.model_classifier = Model(classifier_model_input, classifier)
        self.model_all = Model([img_input, roi_input], rpn + classifier) if(args.mode == 'train') else None

        return self.model_rpn, self.model_classifier, self.model_all
    '''

    def make_model(self, args, base_net):
        model_rpn, model_classifier, model_classifier_only, model_all = None, None, None, None
        img_input = Input(shape=(None, None, 3))
        roi_input = Input(shape=(None, 4))
        shared_layers = base_net.nn_base(img_input, trainable=True)
        rpn = base_net.rpn(shared_layers, self.num_anchors)

        model_rpn = Model(img_input, rpn)

        feature_map_input = Input(shape=(None, None, 1024))
        classifier_only = base_net.classifier(feature_map_input, roi_input, args.num_rois, nb_classes=len(args.class_mapping), trainable=True)
        model_classifier_only = Model([feature_map_input, roi_input], classifier_only)
        model_classifier_only.compile(optimizer='sgd', loss='mae')

        if self.args.test_only:
            model_rpn.compile(optimizer='sgd', loss='mae')

        else :
            classifier = base_net.classifier(shared_layers, roi_input, args.num_rois, nb_classes=len(args.class_mapping), trainable=True)
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

    def train_batch(self, X, Y, rpn_gt_batch):
        loss = np.array([np.Inf]*5)
        rpn_gt_batch += [np.zeros((1, ))]
        loss_rpn = self.rpn.train_on_batch(X, rpn_gt_batch)
        loss[0:2] = loss_rpn[1:3]

        P_rpn = self.rpn.predict_on_batch(X)
        #utility.pickle_save(P_rpn, 'P_rpn_raw.pickle')

        R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], self.args, K.common.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
        #utility.pickle_save(R, 'R_raw.pickle')
        # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
        X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, Y, self.args, self.args.class_mapping)
        #utility.pickle_save([X2, Y1, Y2, IouS], 'cls_sample_raw.pickle')

        num_pos_samples = 0
        if X2 is not None:
            X2, Y1, Y2, num_pos_samples = roi_helpers.get_classifier_samples(X2, Y1, Y2, self.args.num_rois)
            loss_class = self.classifier.train_on_batch([X, X2], [Y1, Y2])
            loss[2:4] = loss_class[1:3]
            loss[-1] = loss[:-1].sum()
        return loss, num_pos_samples

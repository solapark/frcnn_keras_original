import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utility import calc_emb_dist
from epipolar import EPIPOLAR
import cv2

class REID:
    def __init__(self, args):
        self.num_valid_cam = args.num_valid_cam
        self.num_nms = args.num_nms
        self.batch_size = args.batch_size
        self.rpn_stride = args.rpn_stride

        self.reid_min_emb_dist = args.reid_min_emb_dist
        self.cam_idx = np.repeat(np.arange(self.num_valid_cam), self.num_nms).reshape(self.num_valid_cam, self.num_nms, 1)
        self.num_nms_arange = np.arange(self.num_nms)
        self.num_nms_arange_repeat = np.repeat(self.num_nms_arange, self.num_valid_cam).reshape(self.num_nms, self.num_valid_cam, 1).transpose(1, 0, 2)
        self.box_idx_stack = np.concatenate([self.cam_idx, self.num_nms_arange_repeat], 2).reshape(self.num_valid_cam*self.num_nms, 2) #(num_valid_cam*num_nms, 2)

        self.epipolar = EPIPOLAR(args)
 
    def get_min_emb_dist_idx(self, emb, embs, thresh = np.zeros(0), is_want_dist = 0, epi_dist = None): 
        '''
        Args :
            emb (shape : m, n)
            embs (shape : m, k, n)
            thresh_dist : lower thersh. throw away too small dist (shape : m, )
        Return :
            min_dist_idx (shape : m, 1)
        '''
        emb_ref = emb[:, np.newaxis, :]
        dist = calc_emb_dist(emb_ref, embs) #(m, k)
        if epi_dist :
            dist[np.where(epi_dist > self.args.epi_dist_thresh)] = np.inf

        if(thresh.size) : 
            thresh = thresh[:, np.newaxis] #(m, 1)
            dist[dist<=thresh] = np.inf 
        min_dist_idx = np.argmin(dist, 1) #(m, )
        if(is_want_dist):
            min_dist = dist[np.arange(len(dist)), min_dist_idx]
            return min_dist_idx, min_dist
        return min_dist_idx


    def get_ref(self, pred_box_prob, pred_box, pred_box_emb) : 
        pred_box_prob_stack = np.reshape(pred_box_prob, (self.num_valid_cam*self.num_nms, ))
        top_N_box_idx = self.box_idx_stack[pred_box_prob_stack.argsort()[-self.num_nms:]]
        top_N_box_idx = tuple(top_N_box_idx.T)
        ref_cam_idx = top_N_box_idx[0]

        ref_box = pred_box[top_N_box_idx]
        ref_emb = pred_box_emb[top_N_box_idx]

        return ref_cam_idx, ref_box, ref_emb

    def get_ref_cam_idx_batch(self, pred_box_prob_batch, pred_box_batch, pred_box_emb_batch) : 
        ref_cam_idx_batch = []
        for pred_box_prob, pred_box, pred_box_emb in zip(pred_box_prob_batch, pred_box_batch, pred_box_emb_batch) :
            ref_cam_idx, _, _ = self.get_ref(pred_box_prob, pred_box, pred_box_emb)
            ref_cam_idx_batch.append(ref_cam_idx)
        return np.array(ref_cam_idx_batch)
 
    def get_batch(self, *args):
        reid_box_pred_batch, is_valid_batch, dist_batch = [], [], []
        for args_one_batch in zip(*args) :
            reid_box_pred, is_valid, dist = self.get_reid_box(*args_one_batch)
            reid_box_pred_batch.append(reid_box_pred)
            is_valid_batch.append(is_valid)
            dist_batch.append(dist)
        return np.array(reid_box_pred_batch), np.array(is_valid_batch), np.array(dist_batch)

    def get_reid_box(self, pred_box, pred_box_emb, pred_box_prob, extrins):
        """get ref idx, postive idx, negative idx for reid training
            Args : 
                pred_box : x1, y1, x2, y2 #(num_valid_cam, 300, 4)
                pred_box_emb #(num_valid_cam, 300, featsize) 
                pred_box_prob #(num_valid_cam, 300)
                extrins #(num_cam, 3, 3)
            Return :
                reid_box #(300, num_valid_cam, 4)
                is_valid #(300, num_valid_cam)
                dist #(300, num_valid_cam)
                    distance from ref box
        """
        reid_box = np.zeros((self.num_nms, self.num_valid_cam, 4))
        is_valid = np.ones((self.num_nms, self.num_valid_cam))
        dist = np.zeros((self.num_nms, self.num_valid_cam))

        ref_cam_idx, ref_box, ref_emb = self.get_ref(pred_box_prob, pred_box, pred_box_emb)
        reid_box[self.num_nms_arange, ref_cam_idx] = ref_box
        self.epipolar.calc_T_a2b(extrins)
 
        for offset in range(1, self.num_valid_cam):
            target_cam_idx = (ref_cam_idx + offset) % self.num_valid_cam
            cand_emb = pred_box_emb[target_cam_idx]
            cand_box = pred_box[target_cam_idx]

            epi_dist = np.ones((self.num_nms, self.num_nms))
            for i in range(self.num_nms):
                epi_dist[i] = self.epipolar.get_epipolar_dist(ref_cam_idx[i], target_cam_idx[i], ref_box[i].reshape(-1, 4), cand_box[i])

            min_dist_idx, min_dist = self.get_min_emb_dist_idx(ref_emb, cand_emb, is_want_dist=True, epi_dist = epi_dist)
            reid_box[self.num_nms_arange, target_cam_idx] = pred_box[target_cam_idx, min_dist_idx]
            dist[self.num_nms_arange, target_cam_idx] = min_dist

            invalid_idx = np.where(min_dist < self.reid_min_emb_dist)
            invalid_nms_idx = self.num_nms_arange[invalid_idx]
            invalid_target_cam_idx = target_cam_idx[invalid_idx]
            is_valid[invalid_nms_idx, invalid_target_cam_idx] = 0
        return reid_box, is_valid, dist

    def draw_reid_batch(self, box_batch, is_valid_batch, ref_cam_idx_batch, imgs_batch, dist_batch, waitKey=0):
        box_batch = box_batch.astype(int)*self.rpn_stride
        for batch_idx in range(self.batch_size):
            imgs_in_one_batch = imgs_batch[batch_idx]
            boxes_in_one_batch =  box_batch[batch_idx]
            is_valids_in_one_batch =  is_valid_batch[batch_idx]
            ref_cam_idx_in_one_batch = ref_cam_idx_batch[batch_idx]
            dist_in_one_batch = dist_batch[batch_idx]
            img_list = list(imgs_in_one_batch)
            for box_idx in range(self.num_nms) :
                boxes = boxes_in_one_batch[box_idx]
                is_valids = is_valids_in_one_batch[box_idx]
                ref_cam_idx = ref_cam_idx_in_one_batch[box_idx]
                dists = dist_in_one_batch[box_idx]
                result_img_list = []
                for cam_idx in range(self.num_valid_cam):
                    box = boxes[cam_idx] 
                    is_valid = is_valids[cam_idx]
                    dist = dists[cam_idx]

                    if cam_idx == ref_cam_idx :
                        color = (0, 0, 255) 
                    elif is_valid :
                        color = (0, 0, 100)
                    else :
                        color = (0, 0, 0)
                    reuslt_img = draw_box(img_list[cam_idx], box, name = None, color = color, is_show = False, text = '%.2f'%dist)
                    result_img_list.append(reuslt_img)
                concat_img = get_concat_img(result_img_list)
                cv2.imshow('reid', concat_img)
                cv2.waitKey(waitKey)

if __name__ == '__main__':
    from option import args
    import pickle
    with open('/home/sap/frcnn_keras/mv_train_two_reid.pickle', 'rb') as f:
        reid_pickle = pickle.load(f)
    pred_box, pred_box_emb, pred_box_prob, reid_box_gt = reid_pickle
    
    reid = REID(args)
    reid_box_pred, is_valid = reid.get_reid_box(pred_box, pred_box_emb, pred_box_prob)
    print('reid_box_pred.shape', reid_box_pred.shape, 'is_valid', is_valid.shape)
    pred_box_batch, pred_box_emb_batch, pred_box_prob_batch = list(map(lambda a : np.expand_dims(a, 0), [pred_box, pred_box_emb, pred_box_prob]))
    reid_box_pred_batch, is_valid_batch = reid.get_batch(pred_box_batch, pred_box_emb_batch, pred_box_prob_batch)
    print('reid_box_pred_batch.shape', reid_box_pred_batch.shape, 'is_valid_batch', is_valid_batch.shape)
    print(np.array_equal(reid_box_pred_batch[0], reid_box_pred), np.array_equal(is_valid_batch[0], is_valid))

    '''
    is_valid = np.ones((self.num_nms, self.num_valid_cam))
    with open('/home/sap/frcnn_keras/pred_box_is_valid.pickle', 'wb') as f:
        pickle.dump(is_valid, f)

    for i in range(10) :
        print('gt', reid_box_gt[i])
        print('pred', reid_box_pred[i])
        print('valid', is_valid[i])

    if(np.array_equal(reid_box_gt, reid_box_pred)) :
        print('good')
    else :
        print('bad')
    '''

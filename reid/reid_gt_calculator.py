import numpy as np
import cv2
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utility import iou, calc_emb_dist, get_min_emb_dist_idx
import utility
from itertools import permutations

class REID_GT_CALCULATOR:
    def __init__(self, args):
        self.args = args
        self.reid_gt_min_overlap = args.reid_gt_min_overlap
        self.rpn_stride = args.rpn_stride
 
    def get_batch(self, *args):
        ref_pos_neg_idx_batch = []
        for batch_idx, args_one_batch in enumerate(zip(*args)) :
            ref_pos_neg_idx = self.get_ref_pos_neg_idx(*args_one_batch)
            #ref_pos_neg_idx = np.insert(ref_pos_neg_idx, 0, batch_idx, axis=2) 
            ref_pos_neg_idx_batch.append(ref_pos_neg_idx)
        return np.array(ref_pos_neg_idx_batch)
       
    def calc_ious(self, box, boxes):
        '''
        Args :
            box #(4, )
            boxes #(N, 4)
        Return :
            ious #(N, )
        '''
        ious = [iou(box, other_box) for other_box in boxes]
        return np.array(ious)
        
    def get_ref_pos_neg_idx(self, pred_box, pred_box_idx, all_box_emb, gt_insts):
        """get ref idx, postive idx, negative idx for reid training
            Args : 
                pred_box : x1, y1, x2, y2 #(num_cam, 300, 4)
                pred_box_idx : CHWA idx #(num_cam, 300, 4) 
                all_box_emb #(num_cam, H, W, A, featsize) 
                gt_insts #(num_gt_ints, )
            Return :
                ref_pos_neg_idx : CHWA idx of ref, pos, neg  #(3, N, 4)
                    1st axis : ref, pos, neg
                    2nd axis : num sample  
                    3rd axis : CHWA idx of ref, pos , neg
        """
        ref_pos = np.zeros((0, 2, 4), dtype='int')
        for gt_inst in gt_insts :
            ref = np.zeros((0, 4), dtype='int')
            for cam_idx, gt_box in gt_inst['resized_box'].items():
                gt_box = np.array(gt_box)/self.rpn_stride
                #gt_box[[2,1]] = gt_box[[1,2]] #x1, y1, x2, y2
                ious = self.calc_ious(gt_box, pred_box[cam_idx])
                max_idx = np.argmax(ious)
                if(ious[max_idx] > self.reid_gt_min_overlap):
                    CHWA_idx = pred_box_idx[cam_idx, max_idx]
                    ref = np.append(ref, [CHWA_idx], 0)
            ref_perm = np.array(list(permutations(ref, 2)))
            if(len(ref_perm)) : ref_pos = np.append(ref_pos, ref_perm, 0)
            
        ref_pos = ref_pos.transpose((1, 0, 2))
        if(ref_pos.size == 0) : return np.zeros((3, 0, 4))

        num_samples = ref_pos.shape[1]
        if num_samples > self.args.num_max_ven_samples : 
            all_idx = np.arange(num_samples)
            sel_idx = np.random.choice(all_idx, self.args.num_max_ven_samples, replace=False)
            ref_pos = ref_pos[:, sel_idx]

        ref, pos = ref_pos
        ref_CHWA_idx = tuple(ref.T)
        pos_CHWA_idx = tuple(pos.T)
        ref_emb = all_box_emb[ref_CHWA_idx] #(N, feature_size)
        pos_emb = all_box_emb[pos_CHWA_idx] #(N, feature_size)
        pos_dist = calc_emb_dist(ref_emb, pos_emb) #(N, )

        neg_cam_idx = pos_CHWA_idx[0]
        pred_box_emb = all_box_emb[tuple(pred_box_idx.T)].transpose(1, 0, 2)
        neg_cand_emb = pred_box_emb[neg_cam_idx] #(N, 300, feature_size)
        neg_idx = get_min_emb_dist_idx(ref_emb, neg_cand_emb, pos_dist)
        neg_CHWA_idx = pred_box_idx[neg_cam_idx, neg_idx] #(N, 4)
        ref_pos_neg = np.concatenate([ref_pos, np.expand_dims(neg_CHWA_idx, 0)], 0)#(3, N, 4)

        return ref_pos_neg

    def draw_box_from_idx(self, all_boxes, all_images, idx, rpn_stride, name):
        cam_idx, H_idx, W_idx, A_idx = idx
        image = np.copy(all_images[cam_idx]).squeeze()
        box = all_boxes[(cam_idx, H_idx, W_idx, A_idx)]
        box = (box*rpn_stride).astype(int)
        utility.draw_box(image, box, name)

    def draw_anchor_pos_neg(self, R_list, anchor_pos_neg_idx, debug_img) : 
        all_boxes = [R[0] for R in R_list]
        all_boxes = np.stack(all_boxes, 0) #(num_cam, 4, H, W, A)
        all_boxes = all_boxes.transpose(0, 2, 3, 4, 1) #(num_cam, H, W, A, 4)
        anc_idx, pos_idx, neg_idx = anchor_pos_neg_idx[0] #(num_sample, 4)
        for cur_anc_idx, cur_pos_idx, cur_neg_idx in zip(anc_idx, pos_idx, neg_idx) :
            self.draw_box_from_idx(all_boxes, debug_img, cur_anc_idx, self.rpn_stride, 'anchor')
            self.draw_box_from_idx(all_boxes, debug_img, cur_pos_idx, self.rpn_stride, 'pos')
            self.draw_box_from_idx(all_boxes, debug_img, cur_neg_idx, self.rpn_stride, 'neg')
            cv2.waitKey(0)


 
if __name__ == '__main__':
    '''
    pred_box #(num_cam, 300, 4) (x1, y1, x2, y2)
    pred_box_idx : CHWA idx #(num_cam, 300, 4) 
    pred_box_embedding #(num_cam, 300, featsize) 
    gt_insts #(num_gt_ints, )
    '''

    import pickle
    with open('/home/sap/frcnn_keras/mv_train_two_reid_gt.pickle', 'rb') as f:
        result = pickle.load(f)
    pred_box, pred_box_idx, all_box_emb, anchor_pos_neg_idx_gt = result
    
    from dataloader import DATALOADER
    from option import args
    mode = 'train'
    dl = DATALOADER(args, mode)
    _, labels_in_batch = dl[0]
    reid_gt_calculator = REID_GT_CALCULATOR(args)

    pred_box_batch, pred_box_idx_batch, all_box_emb_batch = list(map(lambda a : np.expand_dims(a, 0), [pred_box, pred_box_idx, all_box_emb]))
    ref_pos_neg_idx_batch = reid_gt_calculator.get_batch(pred_box_batch, pred_box_idx_batch, all_box_emb_batch, labels_in_batch)

    print(np.array_equal(ref_pos_neg_idx_batch, anchor_pos_neg_idx_gt))
    print('gt', anchor_pos_neg_idx_gt)
    print('pred', ref_pos_neg_idx_batch)

import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import iou, calc_emb_dist, get_min_emb_dist_idx
from itertools import permutations

class REID_GT_CALCULATOR:
    def __init__(self, args):
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
        
    def get_ref_pos_neg_idx(self, pred_box, view_emb, gt_insts):
        """get ref idx, postive idx, negative idx for reid training
            Args : 
                pred_box : x1, y1, x2, y2 #(num_valid_cam, 300, 4)
                view_emb #(num_valid_cam, 300, featsize) 
                gt_insts #(num_gt_ints, )
            Return :
                ref_pos_neg_idx : idx of ref, pos, neg  #(3, N, 2)
                    1st axis : ref, pos, neg
                    2nd axis : num sample  
                    3rd axis : cam idx, nms idx
        """
        ref_pos = np.zeros((0, 2, 2), dtype='int')
        for gt_inst in gt_insts :
            ref = np.zeros((0, 2), dtype='int')
            for cam_idx, gt_box in gt_inst['resized_box'].items():
                gt_box = np.array(gt_box)/self.rpn_stride
                ious = self.calc_ious(gt_box, pred_box[cam_idx])
                max_idx = np.argmax(ious)
                if(ious[max_idx] > self.reid_gt_min_overlap):
                    ref = np.append(ref, [[cam_idx, max_idx]], 0)
            ref_perm = np.array(list(permutations(ref, 2)))
            if(len(ref_perm)) : ref_pos = np.append(ref_pos, ref_perm, 0)
            
        ref_pos = ref_pos.transpose((1, 0, 2))
        if(ref_pos.size == 0) : return np.zeros((3, 0, 2))
        ref, pos = ref_pos
        ref_idx = tuple(ref.T)
        pos_idx = tuple(pos.T)
        ref_emb = view_emb[ref_idx] #(N, feature_size)
        pos_emb = view_emb[pos_idx] #(N, feature_size)
        pos_dist = calc_emb_dist(ref_emb, pos_emb) #(N, )

        neg_cam_idx = pos_idx[0]
        pred_box_emb = view_emb.transpose(1, 0, 2) #(300, num_valid_cam, feature_size)
        neg_cand_emb = pred_box_emb[neg_cam_idx] #(N, 300, feature_size)
        neg_idx = get_min_emb_dist_idx(ref_emb, neg_cand_emb, pos_dist)
        neg_idx = np.column_stack((neg_cam_idx, neg_idx)) #(N, 2)
        ref_pos_neg = np.concatenate([ref_pos, np.expand_dims(neg_idx, 0)], 0)#(3, N, 2)
        return ref_pos_neg

if __name__ == '__main__':
    '''
    pred_box #(num_valid_cam, 300, 4) (x1, y1, x2, y2)
    pred_box_idx : CHWA idx #(num_valid_cam, 300, 4) 
    pred_box_embedding #(num_valid_cam, 300, featsize) 
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

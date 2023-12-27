import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from epipolar import EPIPOLAR

class EPIPOLAR_FILTER :
    def __init__(self, args) :
        self.num_cam = args.num_cam
        self.epipolar = EPIPOLAR(args)

    def get_batch(self, *args):
        reid_box_pred_batch, is_valid_batch, emb_dist_batch = [], [], []
        for args_one_batch in zip(*args) :
            reid_box_pred, is_valid, emb_dist = self.do(*args_one_batch)
            reid_box_pred_batch.append(reid_box_pred)
            is_valid_batch.append(is_valid)
            emb_dist_batch.append(emb_dist)
        return np.array(reid_box_pred_batch), np.array(is_valid_batch), np.array(emb_dist_batch)

    def do(self, pred_box, is_valid, emb_dist, extrins, debug_imgs) :
        self.epipolar.reset(extrins, debug_imgs)
        pick = []
        for bb in range(len(pred_box)) :
            epi_sat = True
            for ref_cam_idx, (ref_box, ref_is_valid) in enumerate(zip(pred_box[bb], is_valid[bb])):
                if not ref_is_valid : continue
                epipolar_lines = self.epipolar.get_epipolar_lines(ref_cam_idx, ref_box)

                for target_cam_idx, (tar_box, tar_is_valid) in enumerate(zip(pred_box[bb], is_valid[bb])):
                    if ref_cam_idx==target_cam_idx or not tar_is_valid : continue
                    cand_boxes = np.expand_dims(tar_box, 0)
                    epi_box_idx, dist = self.epipolar.get_box_idx_on_epiline(epipolar_lines[target_cam_idx], cand_boxes)
                    #if not len(epi_box_idx[0]): self.epipolar.draw_boxes_with_epiline(ref_cam_idx, ref_box, target_cam_idx, epipolar_lines[target_cam_idx], cand_boxes)

                    if not len(epi_box_idx[0]): 
                        epi_sat=False
                        break

                if not epi_sat : break  

            if epi_sat : pick.append(bb) 

        return pred_box[pick], is_valid[pick], emb_dist[pick]

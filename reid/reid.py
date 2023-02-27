import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import utility
import cv2
from epipolar import EPIPOLAR

class REID:
    def __init__(self, args):
        self.args = args
        self.num_cam = args.num_cam
        self.num_nms = args.num_nms
        self.batch_size = args.batch_size
        self.rpn_stride = args.rpn_stride

        self.reid_min_emb_dist = args.reid_min_emb_dist
        self.cam_idx = np.repeat(np.arange(self.num_cam), self.num_nms).reshape(self.num_cam, self.num_nms, 1)
        self.num_nms_arange = np.arange(self.num_nms)
        self.num_nms_arange_repeat = np.repeat(self.num_nms_arange, self.num_cam).reshape(self.num_nms, self.num_cam, 1).transpose(1, 0, 2)
        self.box_idx_stack = np.concatenate([self.cam_idx, self.num_nms_arange_repeat], 2).reshape(self.num_cam*self.num_nms, 2) #(num_cam*num_nms, 2)

        self.is_use_epipolar = args.is_use_epipolar

        if self.is_use_epipolar :
            self.epipolar = EPIPOLAR(args)
 
    def get_ref(self, pred_box_prob, pred_box, pred_box_emb) : 
        pred_box_prob_stack = np.reshape(pred_box_prob, (self.num_cam*self.num_nms, ))
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
        reid_box_pred_batch, is_valid_batch, emb_dist_batch  = [], [], []
        for args_one_batch in zip(*args) :
            if self.is_use_epipolar :
                reid_box_pred, is_valid, emb_dist = self.get_reid_box_epi_const(*args_one_batch)

            else :
                reid_box_pred, is_valid = self.get_reid_box(*args_one_batch)
            reid_box_pred_batch.append(reid_box_pred)
            is_valid_batch.append(is_valid)
            emb_dist_batch.append(emb_dist)

        return np.array(reid_box_pred_batch), np.array(is_valid_batch), np.array(emb_dist_batch)

    def get_reid_box_epi_const(self, pred_box, pred_box_emb, pred_box_prob, extrins, debug_imgs = None):
        """get ref idx, postive idx, negative idx for reid training
            Args : 
                pred_box : x1, y1, x2, y2 #(num_cam, 300, 4)
                pred_box_emb #(num_cam, 300, featsize) 
                pred_box_prob #(num_cam, 300)
            Return :
                reid_box #(300, num_cam, 4)
                is_valid #(300, num_cam)
        """
        self.epipolar.reset(extrins, debug_imgs)

        reid_box = -np.ones((self.num_nms, self.num_cam, 4))
        is_valid = np.zeros((self.num_nms, self.num_cam))
        emb_dist = -np.ones((self.num_nms, self.num_cam), dtype='float')

        ref_cam_idx_list, ref_box_list, ref_emb_list = self.get_ref(pred_box_prob, pred_box, pred_box_emb)

        for i, (ref_cam_idx, ref_box, ref_emb) in enumerate(zip(ref_cam_idx_list, ref_box_list, ref_emb_list)):
            reid_box[i, ref_cam_idx] = ref_box
            is_valid[i, ref_cam_idx] = 1 

            epipolar_lines = self.epipolar.get_epipolar_lines(ref_cam_idx, ref_box)

            match_box = None
            match_box_cam_idx = None
            match_box_emb_dist = np.inf

            epi_boxes_dict = {}
            epi_embs_dict = {}
            for offset in range(1, self.num_cam):
                target_cam_idx = (ref_cam_idx + offset) % self.num_cam
                cand_boxes = pred_box[target_cam_idx]
                cand_embs = pred_box_emb[target_cam_idx]
                epi_box_idx = self.epipolar.get_box_idx_on_epiline(epipolar_lines[target_cam_idx], cand_boxes)
                #self.epipolar.draw_boxes_with_epiline(ref_cam_idx, ref_box, target_cam_idx, epipolar_lines[target_cam_idx], cand_boxes)

                epi_boxes = cand_boxes[epi_box_idx ]
                epi_embs = cand_embs[epi_box_idx ]

                if not epi_boxes.size : 
                    continue

                min_dist_idx, min_dist = utility.get_min_emb_dist_idx(ref_emb, epi_embs, is_want_dist=True)

                #if min_dist < self.reid_min_emb_dist and min_dist < match_box_emb_dist :
                if min_dist < match_box_emb_dist :
                    match_box = epi_boxes[min_dist_idx]
                    match_box_cam_idx = target_cam_idx
                    match_box_emb_dist = min_dist

                epi_boxes_dict[target_cam_idx] = epi_boxes
                epi_embs_dict[target_cam_idx] = epi_embs

            #if match_box is None  or match_box_emb_dist > self.reid_min_emb_dist :
            if match_box is None :
                #print(match_box_emb_dist)
                #self.draw_fail_reid(debug_imgs, ref_box, ref_cam_idx, match_box, match_box_cam_idx, match_box_emb_dist)
                continue

            reid_box[i, match_box_cam_idx] = match_box
            is_valid[i, match_box_cam_idx] = 2 
            emb_dist[i, match_box_cam_idx] = match_box_emb_dist

            for offset in range(1, self.num_cam):
                target_cam_idx = (ref_cam_idx + offset) % self.num_cam
                if target_cam_idx not in epi_boxes_dict :
                    continue

                if match_box_cam_idx == target_cam_idx :
                    continue

                target_boxes = epi_boxes_dict[target_cam_idx]
                target_embs = epi_embs_dict[target_cam_idx]

                ref_epipolar_line = epipolar_lines[target_cam_idx]
                match_epipolar_line = self.epipolar.get_epipolar_line(match_box_cam_idx, match_box, target_cam_idx)

                valid_idx, dist, cross_pnt, is_valid_inst, is_valid_box = self.epipolar.get_box_idx_on_cross_line(ref_epipolar_line, match_epipolar_line, target_boxes)

                '''
                #if not is_valid_inst :
                #    reid_box[i] = -1
                #    is_valid[i] = 0                    
                #    break
                '''

                if not is_valid_box:
                    #fail_boxes[i, target_cam_idx] = target_boxes
                    #fail_dists[i, target_cam_idx] = dist
                    #self.draw_reid(debug_imgs, reid_box[i], is_valid[i], fail_boxes[i], fail_dists[i], cross_pnt)
                    continue

                valid_boxes = target_boxes[valid_idx]
                valid_embs = target_embs[valid_idx]

                min_dist_idx, min_dist = utility.get_min_emb_dist_idx(ref_emb, valid_embs, is_want_dist=True)

                cross_box = valid_boxes[min_dist_idx]

                reid_box[i, target_cam_idx] = cross_box
                is_valid[i, target_cam_idx] = 3 
                emb_dist[i, target_cam_idx] = min_dist

                #self.draw_reid(debug_imgs, reid_box[i], is_valid[i], fail_boxes[i], fail_dists[i], cross_pnt)

        return reid_box, is_valid, emb_dist
    '''

    def get_reid_box_epi_const(self, pred_box, pred_box_emb, pred_box_prob, extrins, debug_imgs = None):
        """get ref idx, postive idx, negative idx for reid training
            Args : 
                pred_box : x1, y1, x2, y2 #(num_cam, 300, 4)
                pred_box_emb #(num_cam, 300, featsize) 
                pred_box_prob #(num_cam, 300)
            Return :
                reid_box #(300, num_cam, 4)
                is_valid #(300, num_cam)
        """
        self.epipolar.reset(extrins, debug_imgs)

        reid_box = -np.ones((self.num_nms, self.num_cam, 4))
        is_valid = np.zeros((self.num_nms, self.num_cam), dtype='uint8')
        fail_boxes = np.empty((self.num_nms, self.num_cam), dtype=object)
        fail_dists = np.empty((self.num_nms, self.num_cam), dtype=object)

        ref_cam_idx_list, ref_box_list, ref_emb_list = self.get_ref(pred_box_prob, pred_box, pred_box_emb)

        final_reid_box = -np.ones((self.num_nms, self.num_cam, 4))
        final_is_valid = np.zeros((self.num_nms, self.num_cam), dtype='uint8')
        inst_cnt = 0
        for i, (ref_cam_idx, ref_box, ref_emb) in enumerate(zip(ref_cam_idx_list, ref_box_list, ref_emb_list)):
            #reid_box[i, ref_cam_idx] = ref_box
            #is_valid[i, ref_cam_idx] = 1 

            epipolar_lines = self.epipolar.get_epipolar_lines(ref_cam_idx, ref_box)

            match_box = None
            match_box_cam_idx = None
            match_box_emb_dist = np.inf

            epi_boxes_dict = {}
            epi_embs_dict = {}
            for offset in range(1, self.num_cam):
                target_cam_idx = (ref_cam_idx + offset) % self.num_cam
                cand_boxes = pred_box[target_cam_idx]
                cand_embs = pred_box_emb[target_cam_idx]
                epi_box_idx = self.epipolar.get_box_idx_on_epiline(epipolar_lines[target_cam_idx], cand_boxes)
                #self.epipolar.draw_boxes_with_epiline(ref_cam_idx, ref_box, target_cam_idx, epipolar_lines[target_cam_idx], cand_boxes)

                epi_boxes = cand_boxes[epi_box_idx ]
                epi_embs = cand_embs[epi_box_idx ]

                if not epi_boxes.size : 
                    continue

                min_dist_idx, min_dist = utility.get_min_emb_dist_idx(ref_emb, epi_embs, is_want_dist=True)

                #if min_dist < self.reid_min_emb_dist and min_dist < match_box_emb_dist :
                if min_dist < match_box_emb_dist :
                    match_box = epi_boxes[min_dist_idx]
                    match_box_cam_idx = target_cam_idx
                    match_box_emb_dist = min_dist

                epi_boxes_dict[target_cam_idx] = epi_boxes
                epi_embs_dict[target_cam_idx] = epi_embs

            #if match_box is None :
            #    continue

            if  min_dist > self.reid_min_emb_dist :
                reid_box[inst_cnt, ref_cam_idx] = ref_box
                is_valid[inst_cnt, ref_cam_idx] = 1 
                inst_cnt += 1
                if inst_cnt == self.args.num_nms :
                    break

            reid_box[inst_cnt, ref_cam_idx] = ref_box
            is_valid[inst_cnt, ref_cam_idx] = 1 
            reid_box[inst_cnt, match_box_cam_idx] = match_box
            is_valid[inst_cnt, match_box_cam_idx] = 2 

            for offset in range(1, self.num_cam):
                target_cam_idx = (ref_cam_idx + offset) % self.num_cam
                if target_cam_idx not in epi_boxes_dict :
                    continue

                if match_box_cam_idx == target_cam_idx :
                    continue

                target_boxes = epi_boxes_dict[target_cam_idx]
                target_embs = epi_embs_dict[target_cam_idx]

                ref_epipolar_line = epipolar_lines[target_cam_idx]
                match_epipolar_line = self.epipolar.get_epipolar_line(match_box_cam_idx, match_box, target_cam_idx)

                valid_idx, dist, cross_pnt, is_valid_inst, is_valid_box = self.epipolar.get_box_idx_on_cross_line(ref_epipolar_line, match_epipolar_line, target_boxes)

                #if not is_valid_inst :
                #    reid_box[i] = -1
                #    is_valid[i] = 0                    
                #    break

                if not is_valid_box:
                    #fail_boxes[i, target_cam_idx] = target_boxes
                    #fail_dists[i, target_cam_idx] = dist
                    #self.draw_reid(debug_imgs, reid_box[i], is_valid[i], fail_boxes[i], fail_dists[i], cross_pnt)
                    continue

                valid_boxes = target_boxes[valid_idx]
                valid_embs = target_embs[valid_idx]

                min_dist_idx, min_dist = utility.get_min_emb_dist_idx(ref_emb, valid_embs, is_want_dist=True)

                cross_box = valid_boxes[min_dist_idx]

                reid_box[inst_cnt, target_cam_idx] = cross_box
                is_valid[inst_cnt, target_cam_idx] = 3 

                #self.draw_reid(debug_imgs, reid_box[i], is_valid[i], fail_boxes[i], fail_dists[i], cross_pnt)
            inst_cnt += 1
            if inst_cnt == self.args.num_nms :
                break

        return reid_box, is_valid
    '''

    def get_reid_box(self, pred_box, pred_box_emb, pred_box_prob, extrins=None):
        """get ref idx, postive idx, negative idx for reid training
            Args : 
                pred_box : x1, y1, x2, y2 #(num_cam, 300, 4)
                pred_box_emb #(num_cam, 300, featsize) 
                pred_box_prob #(num_cam, 300)
            Return :
                reid_box #(300, num_cam, 4)
                is_valid #(300, num_cam)
        """
        reid_box = np.zeros((self.num_nms, self.num_cam, 4))
        is_valid = np.ones((self.num_nms, self.num_cam))

        ref_cam_idx, ref_box, ref_emb = self.get_ref(pred_box_prob, pred_box, pred_box_emb)
        reid_box[self.num_nms_arange, ref_cam_idx] = ref_box
        
        for offset in range(1, self.num_cam):
            target_cam_idx = (ref_cam_idx + offset) % self.num_cam
            cand_emb = pred_box_emb[target_cam_idx]
            cand_box = pred_box[target_cam_idx]
            min_dist_idx, min_dist = utility.get_min_emb_dist_idx(ref_emb, cand_emb, is_want_dist=True)
            reid_box[self.num_nms_arange, target_cam_idx] = pred_box[target_cam_idx, min_dist_idx]

            invalid_idx = np.where(min_dist < self.reid_min_emb_dist)
            invalid_nms_idx = self.num_nms_arange[invalid_idx]
            invalid_target_cam_idx = target_cam_idx[invalid_idx]
            is_valid[invalid_nms_idx, invalid_target_cam_idx] = 0
        return reid_box, is_valid

    def draw_reid_batch(self, box_batch, is_valid_batch, ref_cam_idx_batch, imgs_batch, waitKey=0):
        box_batch = box_batch.astype(int)*self.rpn_stride
        for batch_idx in range(self.batch_size):
            imgs_in_one_batch = imgs_batch[batch_idx]
            boxes_in_one_batch =  box_batch[batch_idx]
            is_valids_in_one_batch =  is_valid_batch[batch_idx]
            ref_cam_idx_in_one_batch = ref_cam_idx_batch[batch_idx]
            img_list = list(imgs_in_one_batch)
            for box_idx in range(self.num_nms) :
                boxes = boxes_in_one_batch[box_idx]
                is_valids = is_valids_in_one_batch[box_idx]
                ref_cam_idx = ref_cam_idx_in_one_batch[box_idx]
                for cam_idx in range(self.num_cam):
                    box = boxes[cam_idx] 
                    is_valid = is_valids[cam_idx]
                    color = (0, 0, 255) if (cam_idx == ref_cam_idx) else (0, 255, 0) 
                    img_list[cam_idx] = utility.draw_box(img_list[cam_idx], box, color)
            concat_img = utility.get_concat_img(img_list)
            cv2.imshow('reid', concat_img)
            cv2.waitKey(waitKey)


    def draw_fail_reid(self, imgs, ref_box, ref_cam_idx, match_box, match_cam_idx, dist):
        img_list = [np.copy(imgs[i]) for i in range(self.args.num_valid_cam)]

        ref_box = ref_box.astype('int')*self.args.rpn_stride
        match_box = match_box.astype('int')*self.args.rpn_stride

        color = (0, 0, 255)
        text =str(round(dist, 3))

        img_list[ref_cam_idx] = utility.draw_box(img_list[ref_cam_idx], ref_box, '', color, is_show=False)
        img_list[match_cam_idx] = utility.draw_box(img_list[match_cam_idx], match_box, '', color, is_show=False, text = text)

        conc_img = utility.get_concat_img(img_list)
        cv2.imshow('reid_result', conc_img)
        cv2.waitKey(0)    

    def draw_reid(self, imgs, boxes, is_valids, fail_boxes, fail_dists, cross_pnt):
        img_list = []
        boxes = boxes.astype('int')*self.args.rpn_stride
        for cam_idx, (box, is_valid, fail_box_list, fail_dist_list)  in enumerate(zip(boxes, is_valids, fail_boxes, fail_dists)):
            color = (0, 0, 255)
            img = np.copy(imgs[cam_idx])

            if is_valid :
                text =str(int(is_valid))
                img = utility.draw_box(img, box, '', color, is_show=False, text=text)

            else :
                fail_box_list = np.array(fail_box_list, dtype='int')*self.args.rpn_stride
                cross_pnt = self.epipolar.original_pnt_to_resized_pnt(cross_pnt) * self.args.rpn_stride
                cross_pnt = tuple(cross_pnt.astype('int')) 
                for fail_b, fail_d, in zip(fail_box_list, fail_dist_list):
                    text = str(round(fail_d, 3))
                    img = utility.draw_box(img, fail_b, '', color, is_show=False, text=text)
                    img = utility.draw_circle(img, cross_pnt, 10, (255, 0, 0), -1)
                print('cross_pnt', cross_pnt)
                print('fail_box_list', fail_box_list)
                print('fail_dist_list', fail_dist_list)

            img_list.append(img)

        print('is_valids', is_valids)
        print('\n')
        conc_img = utility.get_concat_img(img_list)
        cv2.imshow('reid_result', conc_img)
        cv2.waitKey(0)

import numpy as np
from tqdm import tqdm

from script.json_maker import json_maker
import utility

import argparse 

np.random.seed(0)

if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_json_path', type=str, default='/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority.json')
    parser.add_argument('--dst_json_path', type=str, default='/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority+nms.json')
    parser.add_argument('--num_cam', type=int, default=3)
    parser.add_argument('--nms_thresh', type=float, default=.3)

    args = parser.parse_args()

    cam_ids = [str(i+1) for i in range(args.num_cam)]

    src_json = json_maker([], args.src_json_path, args.num_cam)
    src_json.load()

    dst_json = json_maker([], args.src_json_path, args.num_cam)
    dst_json.load()
    dst_json.path = args.dst_json_path

    scene_name_list = src_json.get_all_scenes()

    for scene_name in tqdm(scene_name_list) :
        dst_json.reset_instance_summary(scene_name)
        dst_json.reset_instances(scene_name)

        inst_clss = list(set(src_json.get_instance_summary(scene_name).values()))

        bboxes = {cls_name : [ [] for _ in range(args.num_cam) ] for cls_name in inst_clss}
        is_valids = {cls_name : [ [] for _ in range(args.num_cam) ] for cls_name in inst_clss}
        probs = {cls_name : [] for cls_name in inst_clss}

        inst_ids = src_json.get_instance_summary(scene_name).keys()

        for inst_id in inst_ids :
            cls = src_json.get_cls_in_instance_summary(scene_name, inst_id)
            valid_cam_id = None
            for cam_idx in range(args.num_cam) :
                cam_id = cam_ids[cam_idx]
                if src_json.is_inst_in_cam(scene_name, cam_id, inst_id) :
                    bboxes[cls][cam_idx].append(src_json.get_inst_box(scene_name, cam_id, inst_id))
                    is_valids[cls][cam_idx].append(1)
                    valid_cam_id= cam_id
                else :
                    bboxes[cls][cam_idx].append([0, 0, 0, 0])
                    is_valids[cls][cam_idx].append(0)
            probs[cls].append(src_json.get_inst_prob(scene_name, valid_cam_id, inst_id))

        inst_idx = 1
        for cls in bboxes:
            cur_bboxes = np.array(bboxes[cls])
            cur_probs = np.array(probs[cls])
            cur_is_valids = np.array(is_valids[cls])

            new_boxes_all_cam, new_probs, new_is_valids_all_cam = utility.non_max_suppression_fast_multi_cam(cur_bboxes, cur_probs, cur_is_valids, overlap_thresh=args.nms_thresh)

            for jk in range(new_boxes_all_cam.shape[1]):
                inst_id = str(inst_idx)
                dst_json.insert_instance_summary(scene_name, inst_id, cls)
                prob = new_probs[jk]
                for cam_idx in range(args.num_cam) : 
                    is_valid = new_is_valids_all_cam[cam_idx, jk]
                    if not is_valid : 
                        continue

                    x1, y1, x2, y2 = new_boxes_all_cam[cam_idx, jk]
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    dst_json.insert_instance(scene_name, cam_ids[cam_idx], inst_id, cls, x1, y1, x2, y2, prob)

                inst_idx += 1

    dst_json.save()

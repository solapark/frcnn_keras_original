import json
import numpy as np
from tqdm import tqdm

from json_maker import json_maker

np.random.seed(0)

src_json_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet.json'
dst_json_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority.json'
num_cam = 3
num_cls = 121

cam_ids = [str(i+1) for i in range(num_cam)]

dst_json = json_maker([], src_json_path, 0)
dst_json.load()

scene_name_list = dst_json.get_all_scenes()
for scene_name in tqdm(scene_name_list) :
    inst_ids = dst_json.get_instance_summary(scene_name).keys()
    for inst_id in inst_ids :
        cls_cnt = np.zeros((num_cls+1, ))
        cls_list = np.zeros((num_cam+1, ))
        prob_list = np.zeros((num_cam+1, ))
        for cam_idx in cam_ids :
            if dst_json.is_inst_in_cam(scene_name, cam_idx, inst_id) :
                cls = dst_json.get_inst_cls(scene_name, cam_idx, inst_id)
                cls_cnt[cls] += 1
                cls_list[int(cam_idx)] = cls
                prob_list[int(cam_idx)] = dst_json.get_inst_prob(scene_name, cam_idx, inst_id)

        if len(np.where(cls_cnt)[0]) != 1 : 
            print(scene_name, inst_id)
            print('\n')

        majority_clss = np.where(cls_cnt==np.max(cls_cnt))
        majority_cls = np.random.choice(majority_clss[0], 1)[0]
        majority_prob = np.mean(prob_list[cls_list == majority_cls])
        majority_cls = majority_cls.item()
        dst_json.insert_instance_summary(scene_name, inst_id, majority_cls)

        for cam_idx in cam_ids :  
            if dst_json.is_inst_in_cam(scene_name, cam_idx, inst_id) :
                dst_json.change_cls_in_cam(scene_name, cam_idx, inst_id, majority_cls)
                dst_json.change_prob_in_cam(scene_name, cam_idx, inst_id, majority_prob)

dst_json.path = dst_json_path
dst_json.save()

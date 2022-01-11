'''
# assign gt id for detection result to evaluate reid performances(ap, fpr-95) which are implemented in ASNet.

input 1. frcnn.json
input 2. gt.json
output. pred.json

for scene_num in scene_nums of frcnn.json
    for cam_num in cam_nums in scene
        pred_insts =  [{'id':, 'subcls':, 'pos':}, ....] 
        gt_inst = [{'id':, 'subcls':, 'pos':}, ....] 
        edges = []
        for pred_inst
            for gt_inst
                wgt = iou(pred_inst['pos'], gt_inst['pos'])
                if wgt : edge.append(['pred_id', 'gt_id', wgt])
        add edges to graph
        maching = matcing result of graph
        new_id = len(gt_insts) +1
        for pred_inst :
            inst_num = matching['pred_id'] if pred_id in matching else new_id
            json.add_inst(scene_num, cam_num, inst_num, subcls, pos)
'''

from tqdm import tqdm

import utility
from bipartite_graph import Bipartite_graph
from json_maker import json_maker

#src_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log.json'
#dst_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log_with_gt_inst_id.json'
#src_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/classification_majority.json'
#dst_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/for_reid_validatioin/classification_majority.json'
#src_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/classification_mvcnn.json'
#dst_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/for_reid_validatioin/classification_mvcnn.json'
#src_path = '/data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05/model_341.json'
#dst_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/for_reid_validatioin/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341.json'
src_path = '/data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning/model_80.json'
dst_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/for_reid_validatioin/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning_model_80.json'

gt_path = '/data1/sap/MessyTable/labels/test.json'

resize_ratio = 0.5555555555555556

if __name__ == '__main__' :
    G = Bipartite_graph()

    src_json = json_maker([], src_path, 0)
    gt_json = json_maker([], gt_path, 0)
    dst_json = json_maker([], dst_path, 0)

    src_json.load()
    gt_json.load()

    dst_json.insert_intrinsics(gt_json.get_intrinsics())
    scene_nums = src_json.get_all_scenes()
    for scene_num in tqdm(scene_nums) :
        instance_summary = gt_json.get_instance_summary(scene_num) 
        instance_ids = [int(id) for id in instance_summary.keys()]
        new_id = max(instance_ids)+1

        dst_json.insert_scene(scene_num)
        scene = src_json.get_scene(scene_num)
        cam_nums = src_json.get_all_cams(scene)
        for cam_num in cam_nums :
            dst_json.insert_cam(scene_num, cam_num)
            src_cam = src_json.get_cam(scene_num, cam_num)
            pred_path, pred_insts = src_json.get_all_inst(src_cam) 
            gt_cam = gt_json.get_cam(scene_num, cam_num)
            gt_path, gt_insts = gt_json.get_all_inst(gt_cam)
            if(pred_path != gt_path):
                print('pred_path', pred_path, 'gt_path', gt_path)
                exit(1)
            dst_json.insert_path(scene_num, cam_num, pred_path)

            extrinsics = gt_json.get_extrinsics(scene_num, cam_num)
            dst_json.insert_extrinsics(scene_num, cam_num, extrinsics)

            edges = []
            for pred_inst in pred_insts :
                for gt_inst in gt_insts :
                    pred_inst_pos = pred_inst['pos']
                    #pred_inst_pos = [p/resize_ratio for p in pred_inst['pos']]
                    wgt = utility.iou(pred_inst_pos, gt_inst['pos'])
                    pred_id_in_G = 'pred_' + pred_inst['inst_id']
                    gt_id_in_G = 'gt_' + gt_inst['inst_id']
                    edges.append((pred_id_in_G, gt_id_in_G, wgt))
        
            G.add_weighted_edges(edges)
            matching = G.match()
            matching = [sorted([k, v], reverse=True) for (k, v) in matching.items()]
            matching = {l[0]:l[1] for l in matching}
            for pred_inst in pred_insts :
                pred_id = pred_inst['inst_id']
                pred_id_in_G = 'pred_' + pred_id
                subcls = pred_inst['subcls']
                x1, y1, x2, y2 = pred_inst['pos']
                prob = pred_inst['prob']
                if pred_id_in_G in matching :
                    gt_id_in_G = matching[pred_id_in_G]
                    dst_id = gt_id_in_G.split('_')[1]
                else :
                    dst_id = str(new_id)
                    new_id += 1
                dst_json.insert_instance(scene_num, cam_num, dst_id, subcls, x1, y1, x2, y2, prob)
                dst_json.insert_pred_id(scene_num, cam_num, dst_id, pred_id)
                #dst_json.insert_instance_summary(scene_num, dst_id, subcls)


    dst_json.sort()
    dst_json.save()

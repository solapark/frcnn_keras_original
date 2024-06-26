'''
# assign gt id to detection result to evaluate reid performances(ap, fpr-95) which are implemented in ASNet.

input 1. det.json
input 2. gt.json
output. det_with_gt_inst_id.json

for scene_id in scene_ids : 
    for cam_id in cam_ids :
        init graph
        pred_insts = frcnn_json.get_insts(scene_id, cam_id)
        gt_insts = gt_json.get_insts(scene_id, cam_id)
        new_id = max(gt_insts inst id) +1

        edges = []
        for pred_inst in pred_insts : 
            for gt_inst in pred_insts : 
                wgt = iou(pred_inst, gt_inst)
                if wgt : 
                    edge.append(['pred_inst_id', 'gt_inst_id', wgt])
        add edges to graph
        maching = matcing result of graph

        for pred_inst in pred_insts:
            if pred_inst_id in matching :
                inst_id = matching['pred_inst_id']
            else :
                inst_id = new_id 
                new_id += 1
            frcnn_with_gt_inst_id.add_inst(scene_id, cam_id, inst_id, subcls, pos)
'''

from tqdm import tqdm

import utility
from bipartite_graph import Bipartite_graph
from json_maker import json_maker

import argparse 
from collections import defaultdict

#src_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log.json'
#dst_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log_with_gt_inst_id.json'

#src_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/classification_majority.json'
#dst_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/for_reid_validatioin/classification_majority.json'
#src_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/classification_mvcnn.json'
#dst_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/for_reid_validatioin/classification_mvcnn.json'
#src_path = '/data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05/model_341.json'
#dst_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/for_reid_validatioin/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341.json'
#src_path = '/data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning/model_80.json'
#dst_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/for_reid_validatioin/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning_model_80.json'

#src_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet.json'
#dst_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/for_reid_validatioin/svdet+asnet.json'

#src_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+triplenet.json'
#dst_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/for_reid_validatioin/svdet+triplenet.json'

#src_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority.json'
#dst_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/for_reid_validatioin/svdet+asnet+majority.json'

#src_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+majority+nms.json'
#dst_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/for_reid_validatioin/svdet+asnet+majority+nms.json'

#src_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+mvcnn.json'
#dst_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/for_reid_validatioin/svdet+asnet+mvcnn.json'

#src_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet+mvcnn+nms.json'
#dst_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/for_reid_validatioin/svdet+asnet+mvcnn+nms.json'

#src_path = '/data3/sap/frcnn_keras_original/experiment/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning/model_80.json'
#dst_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/for_reid_validatioin/mv_messytable_classifier_only_strict_pos_max_dist_epiline_to_box.05_model_341_fine_tunning_model_80.json'

#gt_path = '/data1/sap/MessyTable/labels/test.json'

#resize_ratio = 0.5555555555555556

def find_additional_matches(G, matching):
    additional_matching = defaultdict(list)
    for pred_x in G.nodes():
        if pred_x.startswith('pred_') and  pred_x not in matching:
            max_iou = -1
            best_gt_y = None
            for gt_y in G.neighbors(pred_x):
                weight = G[pred_x][gt_y]['weight']
                if weight > max_iou  :
                    max_iou = weight
                    best_gt_y = gt_y
            if best_gt_y:
                additional_matching[best_gt_y].append(pred_x)
    return additional_matching


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, default='/data1/sap/MessyTable/labels/test.json')
    parser.add_argument('--src_path', type=str, default='/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log.json')
    parser.add_argument('--dst_path', type=str, default='/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet.json')
    parser.add_argument("--save_core", action="store_true", default=False)
    parser.add_argument("--assign_gt_class", action="store_true", default=False)
    parser.add_argument("--save_gt_pos", action="store_true", default=False)
    parser.add_argument("--iou_thresh", type=float, default=0.)
    parser.add_argument("--multiple_pred", action="store_true", default=False)
    parser.add_argument("--one2N", action="store_true", default=False)
        

    args = parser.parse_args()


    src_json = json_maker([], args.src_path, 0)
    src_json.load()
    gt_json = json_maker([], args.gt_path, 0)
    gt_json.load()

    if args.one2N :
        dst_json = json_maker([], args.gt_path, 0) 
        dst_json.load()
        dst_json.path = args.dst_path
    else : 
        dst_json = json_maker([], args.dst_path, 0) 
        dst_json.insert_intrinsics(gt_json.get_intrinsics())

    scene_ids = src_json.get_all_scenes()
    for scene_id in tqdm(scene_ids) :
        #G = Bipartite_graph()
        instance_summary = gt_json.get_instance_summary(scene_id) 
        instance_ids = [int(id) for id in instance_summary.keys()]
        new_id = max(instance_ids)+1

        if not args.one2N : dst_json.insert_scene(scene_id)
        scene = src_json.get_scene(scene_id)
        cam_ids = src_json.get_all_cams(scene)
        for cam_id in cam_ids :
            G = Bipartite_graph()
            if not args.one2N : dst_json.insert_cam(scene_id, cam_id)

            src_cam = src_json.get_cam(scene_id, cam_id)
            pred_path, pred_insts = src_json.get_all_inst(src_cam) 

            gt_cam = gt_json.get_cam(scene_id, cam_id)
            gt_path, gt_insts = gt_json.get_all_inst(gt_cam)

            if(pred_path != gt_path):
                #print('pred_path', pred_path, 'gt_path', gt_path)
                exit(1)

            if not args.one2N : dst_json.insert_path(scene_id, cam_id, pred_path)

            extrinsics = gt_json.get_extrinsics(scene_id, cam_id)
            if not args.one2N : dst_json.insert_extrinsics(scene_id, cam_id, extrinsics)

            edges = []
            for pred_inst in pred_insts :
                for gt_inst in gt_insts :
                    #pred_inst_pos = [p/resize_ratio for p in pred_inst['pos']]
                    #wgt = utility.iou(pred_inst_pos, gt_inst['pos'])
                    wgt = utility.iou(pred_inst['pos'], gt_inst['pos'])
                    pred_id_in_G = 'pred_' + pred_inst['inst_id']
                    gt_id_in_G = 'gt_' + gt_inst['inst_id']
                    if wgt >= args.iou_thresh : edges.append((pred_id_in_G, gt_id_in_G, wgt))
        
            if args.one2N :
                edges_sorted = sorted(edges, key=lambda x: x[2]) 
                matching = {pred_id:[gt_id, iou] for pred_id, gt_id, iou in edges_sorted} #if a pred belong to multiple gt, choose gt with biggest iou.
                matching = [(gt_id, pred_id, iou) for pred_id, (gt_id, iou) in matching.items()]
                
                for gt_id, pred_id, iou in matching[::-1]:
                    g = gt_id.split('_')[1]
                    p = int(pred_id.split('_')[1])
                    x1, y1, x2, y2 = pred_insts[p]['pos']
                    prob = pred_insts[p]['prob']
                    dst_json.insert_rp(scene_id, cam_id, g, p, x1, y1, x2, y2, iou, prob)

            else :
                G.add_weighted_edges(edges)
                matching = G.match()
                matching = [sorted([k, v], reverse=True) for (k, v) in matching.items()]
                matching = {l[0]:l[1] for l in matching}
                additional_pred_ids = find_additional_matches(G.G, matching) if args.multiple_pred else None

                for pred_inst in pred_insts :
                    pred_id = pred_inst['inst_id']
                    pred_id_in_G = 'pred_' + pred_id
                    x1, y1, x2, y2 = gt_inst['pos'] if args.save_gt_pos else pred_inst['pos']
                    prob = pred_inst['prob']
                    if pred_id_in_G in matching :
                        gt_id_in_G = matching[pred_id_in_G]
                        dst_inst_id = gt_id_in_G.split('_')[1]
                    else :
                        if args.save_core :
                            continue
                        dst_inst_id = str(new_id)
                        new_id += 1

                    if args.assign_gt_class and gt_json.is_inst_in_instance_summary(scene_id, dst_inst_id) : 
                        subcls = gt_json.get_inst_cls(scene_id, cam_id, dst_inst_id) 
                    else :
                        subcls = pred_inst['subcls']

                    if args.multiple_pred :
                        pred_id = [pred_id] + [pred_id_str.split('_')[1] for pred_id_str in additional_pred_ids]
                        pred_id.sort()

                    dst_json.insert_instance(scene_id, cam_id, dst_inst_id, subcls, x1, y1, x2, y2, prob)
                    dst_json.insert_pred_id(scene_id, cam_id, dst_inst_id, pred_id)

                    if not dst_json.is_inst_in_instance_summary(scene_id, dst_inst_id) : 
                        dst_json.insert_instance_summary(scene_id, dst_inst_id, subcls)

    dst_json.sort()
    dst_json.save()

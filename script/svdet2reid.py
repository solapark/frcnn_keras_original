import networkx as nx
import json
from itertools import combinations 
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
from tqdm import tqdm

from bipartite_graph import Bipartite_graph
from json_maker import json_maker

asnet_result_path = '/data3/sap/Messytable/models/asnet/results_img_pairs.json'

src_json_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log_with_gt_inst_id.json'
dst_json_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/reid.json'
num_cam = 3

scale_up_factor = 10
scale_down_factor = 50

def get_id(cam_idx, inst_id):
    return cam_idx + '_' + inst_id

def split_id(id):
    return id.split('_')

def get_box(results_to_save, scene_name, main_cam_idx, inst_id):
    sec_cam_idx = (cam_idx + 1) % num_cam 
    key = ','.join([scene_name, main_cam_idx, sec_cam_idx])
    inst_id_idx = results_to_save[key]['main_bbox_id'].index(inst_id)
    box = results_to_save[key]['main_bbox_pos'][inst_id_idx]
    
def add_node(G, cam_idx, bbox_ids):
    for bbox_id in bbox_ids :
        id = get_id(cam_idx, bbox_id)
        G.add_node(id)

def get_img_path(scene_name, cam_idx):
    return scene_name + str('-0') + cam_idx + '.jpg'

def scale_data(x):
    # 1D data , scale (0, 1)
    scaler = MinMaxScaler()
    scaler.fit(x)
    return scaler.transform(x)

src_json = json_maker([], src_json_path, 0)
src_json.load()

dst_json = json_maker([], dst_json_path, 0)

with open(asnet_result_path, 'r') as file:
    asnet_result = json.load(file)

scene_list = list(asnet_result.keys())
offset = len(list(combinations(range(num_cam), 2)))
scene_list = [scene_list[i:i+offset] for i in range(0, len(scene_list), offset)]

for scene_cam_pairs in tqdm(scene_list) :
    G = nx.Graph()
    cam_mask = np.zeros((num_cam, ))
    for scene_cam_pair in scene_cam_pairs :
        scene_name, main_cam_idx, sec_cam_idx = scene_cam_pair.split(',')

        bi_G = Bipartite_graph()
        content = asnet_result[scene_cam_pair]

        app_dist_np = np.array(content['app_dist'], dtype='float').reshape(-1, 1)
        epi_dist_np = np.array(content['epi_dist'], dtype='float').reshape(-1, 1)
        overall_dist_np = np.add(epi_dist_np * scale_up_factor, app_dist_np) / scale_down_factor
        score = 1 - scale_data(overall_dist_np)
        
        main_bbox_ids = np.array(content['main_bbox_id'])
        sec_bbox_ids = np.array(content['sec_bbox_id'])

        if not cam_mask[int(main_cam_idx)-1] :
            add_node(G, main_cam_idx, main_bbox_ids)
            cam_mask[int(main_cam_idx)-1] = 1

        if not cam_mask[int(sec_cam_idx)-1] :
            add_node(G, sec_cam_idx, sec_bbox_ids)
            cam_mask[int(sec_cam_idx)-1] = 1

        score = score.reshape((len(main_bbox_ids), len(sec_bbox_ids)))

        valid_idx = np.where(score >= .5)
        valid_main_bbox_ids = main_bbox_ids[valid_idx[0]]
        valid_sec_bbox_ids = sec_bbox_ids[valid_idx[1]]
        valid_score = score[valid_idx]

        edges = []
        for main_bbox_id, sec_bbox_id, s in zip(valid_main_bbox_ids, valid_sec_bbox_ids, valid_score) :
            main_id = get_id(main_cam_idx, main_bbox_id)
            sec_id = get_id(sec_cam_idx, sec_bbox_id)
            edges.append((main_id, sec_id, s))
        bi_G.add_weighted_edges(edges)

        matching = bi_G.match()
        for (k, v) in matching.items():
            G.add_edge(k, v)

    reids = []
    for src_node in G.nodes():
        src_cam_idx, _ = split_id(src_node)
        reid_cands = []
        for dst_node in G.nodes() :
            dst_cam_idx, _= split_id(dst_node)
            if not src_cam_idx == dst_cam_idx :
                all_paths = nx.all_simple_paths(G, source=src_node, target=dst_node) 
                for path in all_paths :
                    reid_cands.append(path)

        for reid_cand in reid_cands :
            is_cam_visited = np.zeros((num_cam, ))
            cur_reid = set()
            for id in reid_cand :
                cam_idx, _ = split_id(id)
                if not is_cam_visited[int(cam_idx)-1] :
                    cur_reid.add(id)
                    is_cam_visited[int(cam_idx)-1] = 1
                else :
                    break

            is_valid = True
            for r in reids :
                if r >= cur_reid:
                    is_valid = False
                    break

                elif r < cur_reid :
                    reids.remove(r)
                    break

            if is_valid :
                reids.append(cur_reid)
            
    dst_json.insert_scene(scene_name)
    for cam_idx in range(1, num_cam+1) :
        cam_idx = str(cam_idx)
        dst_json.insert_cam(scene_name, cam_idx)
        img_path = get_img_path(scene_name, cam_idx)
        dst_json.insert_path(scene_name, cam_idx, img_path)

    for i, r in enumerate(reids) :
        new_inst_id = str(i+1)
        subcls = 120
        dst_json.insert_instance_summary(scene_name, new_inst_id, subcls)
        for id in r :
            cam_idx, inst_id = split_id(id)
            cls = src_json.get_inst_cls(scene_name, cam_idx, inst_id)
            x1, y1, x2, y2 = src_json.get_inst_box(scene_name, cam_idx, inst_id)
            prob = src_json.get_inst_prob(scene_name, cam_idx, inst_id)
            dst_json.insert_instance(scene_name, cam_idx, new_inst_id, cls, x1, y1, x2, y2, prob)
    
dst_json.save()



'''
for scene in scenes :
    for cam_pair in cam_pairs :
        for inst_pair in inst_pairs :
            if score > 0.5 :
                add to bipar graph
        add bipar_graph_matching_pairs to graph

    reid = []
    for v in V of graph :
        dfs = dfs(v)
        for branch in dfs :
            mask = [0, 0, 0]
            mask[cam_idx of v] = 1
            cur_reid = {}
            for box in branch :
                if not mask :
                    cur_reid.add(cam_idx of box)
                    mask[cam_idx of box] = 1
                else :
                    break

            for r in reid :
                intersection = r and cur_reid  
                if intersection == cur_reid :
                    continue
                elif intersection == r :
                    reid.remove(r)
                    reid.add(cur_reid)
                else : 
                    reid.add(cur_reid)

    for i, r in enumerate(reid) :
        add_instance 
        for box in r :
            add_inst(box)
'''

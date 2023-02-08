from tqdm import tqdm

import utility
from json_maker import json_maker

import argparse 

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, default='/data1/sap/MessyTable/labels/test.json')
    parser.add_argument('--src_path', type=str, default='/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet_gt_aligned.json')
    parser.add_argument('--dst_path', type=str, default='/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet_gt_aligned.json')

    args = parser.parse_args()

    gt_json = json_maker([], args.gt_path, 0)
    dst_json = json_maker([], args.src_path, 0)

    dst_json.load()
    gt_json.load()

    dst_json.path = args.dst_path

    scene_ids = dst_json.get_all_scenes()
    for scene_id in tqdm(scene_ids) :
        scene = dst_json.get_scene(scene_id)
        cam_ids = dst_json.get_all_cams(scene)
        for cam_id in cam_ids :
            corners = gt_json.get_corners(scene_id, cam_id)
            dst_json.insert_corners(scene_id, cam_id, corners)

    dst_json.save()

import argparse 
import os

from simple_label import Simple_label
from json_maker import json_maker 
import utils

if __name__ == '__main__' :
    parser=argparse.ArgumentParser()
    parser.add_argument('--json_path', default = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log.json')
    #parser.add_argument('--json_path', default = '/data3/sap/frcnn_keras_original/experiment/debug_backup/mv_messytable_fine_tunning/test_model_21.json')
    parser.add_argument('--simple_label_dir', default= '/data3/sap/mAP/input/detection-results')
    #parser.add_argument('--json_path', default = '/data1/sap/MessyTable/labels/test.json')
    #parser.add_argument('--simple_label_dir', default= '/data3/sap/mAP/input/ground-truth')
    parser.add_argument('--img_base_path', default = '/data1/sap/MessyTable/images')
    parser.add_argument('--num_cam', type=int, default = 3)

    args = parser.parse_args()

    json = json_maker([], args.json_path, 0)
    json.load()

    file_name = utils.get_value_in_pattern(args.json_path, '.*/(.*).json')
    is_gt = 1 if file_name in ['train', 'val', 'test', 'val_easy', 'val_medium', 'val_hard', 'test_easy', 'test_medium', 'test_hard'] else 0

    all_scene_nums = json.get_all_scenes() 
    for i, scene_num in enumerate(all_scene_nums) :
        scene = json.get_scene(scene_num)
        all_cam_idx = json.get_all_cams(scene) 

        for cam_idx in all_cam_idx :
            if int(cam_idx) > args.num_cam : continue
            cam = scene['cameras'][cam_idx]
            image_name, instances = json.get_all_inst(cam)
            txt_name = image_name.split('.')[0] + '.txt'
            save_path = os.path.join(args.simple_label_dir, txt_name)
            simple_label = Simple_label(save_path, [], args.img_base_path)
            labels = simple_label.get_simple_label(image_name, instances) 
            if is_gt :
                labels = [[int(cls_name)-1, x1, y1, x2, y2] for (_, x1, y1, x2, y2, cls_name, prob) in labels]
            else :
                labels = [[cls_name, prob, x1, y1, x2, y2] for (_, x1, y1, x2, y2, cls_name, prob) in labels]
            

            simple_label.write_txt(labels)

        #if i == 0 : break


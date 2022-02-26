import argparse 

from simple_label import Simple_label
from json_maker import json_maker 

if __name__ == '__main__' :
    parser=argparse.ArgumentParser()
    parser.add_argument('--json_path', default = '/data1/sap/MessyTable/labels/test.json')
    parser.add_argument('--simple_label_path', default= '/data1/sap/MessyTable/labels/test.txt')
    parser.add_argument('--img_base_path', default = '/data1/sap/MessyTable/images')
    parser.add_argument('--num_cam', type=int, default = 3)

    args = parser.parse_args()


    json = json_maker([], args.json_path, 0)
    json.load()
    simple_label = Simple_label(args.simple_label_path, [], args.img_base_path)

    all_scene_nums = json.get_all_scenes() 
    for scene_num in all_scene_nums :
        scene = json.get_scene(scene_num)
        all_cam_idx = json.get_all_cams(scene) 

        for cam_idx in all_cam_idx :
            if int(cam_idx) > args.num_cam : continue
            cam = scene['cameras'][cam_idx]
            path_name, instances = json.get_all_inst(cam)
            labels = simple_label.get_simple_label(path_name, instances) 
            simple_label.write(labels)


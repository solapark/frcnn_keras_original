from simple_label import Simple_label
from json_maker import json_maker 

json_path = '/data1/sap/MessyTable/labels/test.json'
simple_label_path = '/data1/sap/MessyTable/labels/test.txt'
class_list = []
img_base_path = '/data1/sap/MessyTable/images'

if __name__ == '__main__' :
    json = json_maker([], json_path, 0)
    json.load()
    simple_label = Simple_label(simple_label_path, class_list, img_base_path)

    all_scene_nums = json.get_all_scenes() 
    for scene_num in all_scene_nums :
        scene = json.get_scene(scene_num)
        all_cam_idx = json.get_all_cams(scene) 

        for cam_idx in all_cam_idx :
            cam = scene['cameras'][cam_idx]
            path_name, instances = json.get_all_inst(cam)
            labels = simple_label.get_simple_label(path_name, instances) 
            simple_label.write(labels)


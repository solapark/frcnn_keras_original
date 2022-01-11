'''
json
    - intrinsics
    - scenes
        - (scene_name)
            - instance_summary
                - (instnace_num) : (cls)
            - cameras
                - 1
                    - pathname : (pathname)
                    - extrinsics
                    - corners
                    - instances
                        - (instance_num)
                            - subcls
                            - cls : (cls)
                            - pos
                                - 0 : (x1)
                                - 1 : (y1)
                                - 2 : (x2)
                                - 3 : (y2)
                - 2
                - 3
'''

from utils import csv2list, get_file_list_from_dir, check_pattern_exist, replace, split_list
from json_maker import json_maker
from yolo_label import Yolo_label
import os

csv_path = '/home/sap/frcnn_keras/script/file_pattern.csv'
img_dir = '/data1/sap/interpark_devkit/data/Images'
label_dir = '/data3/sap/INTERPARK/train/extra_1210_mod/labels'
json_base_path = '/data3/sap/frcnn_keras/data/INTERPARK/'
cls_list = ['myzzo', 'tunacan']
               
val_ratio = .2
test_ratio = .2
train_ratio = 1-val_ratio-test_ratio

all_json_path = os.path.join(json_base_path, 'all.json')
train_json_path = os.path.join(json_base_path, 'train_backup.json')
val_json_path = os.path.join(json_base_path, 'val.json')
test_json_path = os.path.join(json_base_path, 'test.json')

def copy_scenes(src_jm, dst_jm, scenes):
    for scene in scenes :
        dst_jm.insert_scene(scene, src_jm.get_scene(scene))

if __name__ == '__main__' : 
    jm  = json_maker([], all_json_path, 0)
    yolo_label = Yolo_label(img_dir, label_dir, cls_list)
    img_names = get_file_list_from_dir(img_dir, is_full_path=False)
    seen_patterns = []
    #2. for line : add instance summary and instances
    lines = csv2list(csv_path, header = False)
    for line in lines :
        cls, pattern, cam_num = line
        cls = cls_list.index(cls)
        valid_cams = list(map(int, cam_num.split('/')))
        for i, img_name in enumerate(img_names) :
            if not check_pattern_exist(img_name, pattern) : continue
            cur_cls, cur_cam_num, cur_scene = replace(pattern, '\g<class> \g<cam_num> \g<scene>', img_name).split()
            cam_num = int(cur_cam_num) 
            if cam_num not in valid_cams: continue
            cur_pattern = '%s_%s'%(cur_cls, cur_scene)
            if cur_pattern not in seen_patterns :
                scene_num = len(seen_patterns)
                scene_num = "%08d"%(scene_num)
                seen_patterns.append(cur_pattern)
                jm.insert_scene(scene_num)
                jm.insert_instance_summary(scene_num, 0, cls)
            else :
                scene_num = seen_patterns.index(cur_pattern)
                scene_num = "%08d"%(scene_num)
            jm.insert_cam(scene_num, cam_num)
            labels = yolo_label.get_labels(img_name)
            _, [x1, y1, x2, y2] = labels[0]
            jm.insert_instance(scene_num, cam_num, 0, cls, x1, y1, x2, y2) 
            img_path = os.path.join(img_dir, img_name)
            jm.insert_path(scene_num, cam_num, img_path)
            #jm.print()
            
    jm.sort()
    #jm.print()
    jm.save()

    train_jm  = json_maker([], train_json_path, 0)
    val_jm  = json_maker([], val_json_path, 0)
    test_jm  = json_maker([], test_json_path, 0)

    num_cam = len(valid_cams)
    all_scenes = jm.get_all_scenes()
    num_all_scenes = len(all_scenes)
    val_size = int(num_all_scenes * val_ratio)
    test_size = int(num_all_scenes * test_ratio)
    train_size = num_all_scenes - val_size - test_size

    train_val_scenes, test_scenes = split_list(all_scenes, train_size+val_size)
    train_scenes, val_scenes = split_list(train_val_scenes, train_size)
 
    copy_scenes(jm, train_jm, train_scenes)
    copy_scenes(jm, val_jm, val_scenes)
    copy_scenes(jm, test_jm, test_scenes)

    all_jm = [train_jm, val_jm, test_jm]
    for jm in all_jm :
        jm.sort()
        jm.save()

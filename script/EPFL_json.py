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

from utils import split_list, csv2list
from json_maker import json_maker

num_frame = 242
num_cam = 6
train_size = 170
val_size = 30
test_size = 42
csv_path = '/data3/sap/EPFL_MVMC/instance_label.csv'
image_base_path = '/data3/sap/EPFL_MVMC/image/c%d/%s.jpg'
train_json_path = '/data3/sap/EPFL_MVMC/train.json'
val_json_path = '/data3/sap/EPFL_MVMC/val.json'
test_json_path = '/data3/sap/EPFL_MVMC/test.json'
class_map = {'bus':0, 'car':1, 'person':2}
               
if __name__ == '__main__' : 
    #1. divide scene to train/val/test
    whole_frames = list(range(1, num_frame+1))
    train_val_frames, test_frames = split_list(whole_frames, train_size+val_size)
    train_frames, val_frames = split_list(train_val_frames, train_size)
    all_frames = [train_frames, val_frames, test_frames]
    for i in range(3): all_frames[i] = ["%08d"%int(frame) for frame in all_frames[i]]
    train_frames, val_frames, test_frames = all_frames

    #2. make train/val/test json with scenes, cameras, path_name
    train_json  = json_maker(train_frames, train_json_path, num_cam)
    val_json  = json_maker(val_frames, val_json_path, num_cam)
    test_json  = json_maker(test_frames, test_json_path, num_cam)
    all_json = [train_json, val_json, test_json]
    
    #2. for line : add instance summary and instances
    lines = csv2list(csv_path, header = False)
    for line in lines :
        frame, cam, cls, obj, x1, y1, x2, y2, inst = line 
        if(inst == '-1') : continue
        frame = "%08d"%int(frame)
        cam = str(int(cam) + 1)
        cls_num = class_map[cls]
        x1, y1, x2, y2 = list(map(int, [x1, y1, x2, y2]))
        for cur_json in all_json :
            if cur_json.is_scene_in_tree(frame) :
                if not cur_json.is_inst_in_instance_summary(frame, inst) :        
                    cur_json.insert_instance_summary(frame, inst, cls_num)
                cur_json.insert_instance(frame, cam, inst, cls_num, x1, y1, x2, y2)
    
    for cur_json in all_json :
        cur_json.sort()
        #cur_json.print()
        cur_json.save()

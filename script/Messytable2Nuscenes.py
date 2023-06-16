import argparse 
from scipy.spatial.transform import Rotation as R
import secrets
import numpy as np
import os
import pickle

from json_maker import json_maker 

CLASS = {'water1': 1, 'water2': 2, 'pepsi': 3, 'coca1': 4, 'coca2': 5, 'coca3': 6, 'coca4': 7, 'tea1': 8, 'tea2': 9, 'yogurt': 10, 'ramen1': 11, 'ramen2': 12, 'ramen3': 13, 'ramen4': 14, 'ramen5': 15, 'ramen6': 16, 'ramen7': 17, 'juice1': 18, 'juice2': 19, 'can1': 20, 'can2': 21, 'can3': 22, 'can4': 23, 'can5': 24, 'can6': 25, 'can7': 26, 'can8': 27, 'can9': 28, 'ham1': 29, 'ham2': 30, 'pack1': 31, 'pack2': 32, 'pack3': 33, 'pack4': 34, 'pack5': 35, 'pack6': 36, 'snack1': 37, 'snack2': 38, 'snack3': 39, 'snack4': 40, 'snack5': 41, 'snack6': 42, 'snack7': 43, 'snack8': 44, 'snack9': 45, 'snack10': 46, 'snack11': 47, 'snack12': 48, 'snack13': 49, 'snack14': 50, 'snack15': 51, 'snack16': 52, 'snack17': 53, 'snack18': 54, 'snack19': 55, 'snack20': 56, 'snack21': 57, 'snack22': 58, 'snack23': 59, 'snack24': 60, 'green_apple': 61, 'red_apple': 62, 'tangerine': 63, 'lime': 64, 'lemon': 65, 'yellow_quince': 66, 'green_quince': 67, 'white_quince': 68, 'fruit1': 69, 'fruit2': 70, 'peach': 71, 'banana': 72, 'fruit3': 73, 'pineapple': 74, 'fruit4': 75, 'strawberry': 76, 'cherry': 77, 'red_pimento': 78, 'green_pimento': 79, 'carrot': 80, 'cabbage1': 81, 'cabbage2': 82, 'eggplant': 83, 'bread': 84, 'baguette': 85, 'sandwich': 86, 'hamburger': 87, 'hotdog': 88, 'donuts': 89, 'cake': 90, 'onion': 91, 'marshmallow': 92, 'mooncake': 93, 'shirimpsushi': 94, 'sushi1': 95, 'sushi2': 96, 'big_spoon': 97, 'small_spoon': 98, 'fork': 99, 'knife': 100, 'big_plate': 101, 'small_plate': 102, 'bowl': 103, 'white_ricebowl': 104, 'blue_ricebowl': 105, 'black_ricebowl': 106, 'green_ricebowl': 107, 'black_mug': 108, 'gray_mug': 109, 'pink_mug': 110, 'green_mug': 111, 'blue_mug': 112, 'blue_cup': 113, 'orange_cup': 114, 'yellow_cup': 115, 'big_wineglass': 116, 'small_wineglass': 117, 'glass1': 118, 'glass2': 119, 'glass3': 120, 'bg':121}

def fill_infos(all_scene_ids, json):
    infos = []
    camera_types = [
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
    ]

    intrinsics = json.get_intrinsics()

    for scene_id in all_scene_ids :
        instance_summary = json.get_instance_summary(scene_id)
        num_inst = len(instance_summary)
        inst_ids = list(instance_summary.keys())
        info = {
            'scene_id': scene_id,
            'scene_token': secrets.token_hex(nbytes=16),
            'cams': dict(),
            'valid_flags': np.ones((num_inst,)),
            'gt_ids' : inst_ids, 
            'gt_names' : None, #inst classes
            'cam_instances' : dict(),
            'cam_instances_valid_flags' : dict()
        }

        gt_names = [class_names[class_ids.index(cls)] for cls in instance_summary.values()]
        info.update(gt_names = gt_names)

        scene = json.get_scene(scene_id)
        all_cam_idx = json.get_all_cams(scene) 

        for cam_id in all_cam_idx:
            intrinsic = intrinsics[cam_id]
            K = np.eye(4)
            K[:3, :3] = np.array(intrinsic).reshape(3, 3)

            extrinsic = json.get_extrinsics(scene_id, cam_id)
            E = np.eye(4)
            E[0:3, 0:3] = R.from_euler('xyz', extrinsic[3:]).as_matrix()
            E[0:3, 3] = np.array(extrinsic[:3])

            world2img = K @ E

            cam = scene['cameras'][cam_id]
            path_name, instances = json.get_all_inst(cam)

            cam_info = {
                'img_path' : os.path.join(args.img_dir, path_name), 
                'type' : cam_id, 
                'intrinsic' : K,
                'extrinsic' : E, 
                'world2img' : world2img, 
            }

            info['cams'].update({cam_id: cam_info})

            info['cam_instances'][cam_id] = np.zeros((num_inst, 4))
            info['cam_instances_valid_flags'][cam_id] = np.zeros((num_inst, ))

            for instance in instances :
                inst_id = instance['inst_id']
                inst_idx = inst_ids.index(inst_id)
                info['cam_instances'][cam_id][inst_idx] = instance['pos']
                info['cam_instances_valid_flags'][cam_id][inst_idx] = 1

        #print(info)
        infos.append(info)

    return infos

if __name__ == '__main__' :
    parser=argparse.ArgumentParser()
    #parser.add_argument('--json_path', default = '/data1/sap/MessyTable/labels/train.json')
    #parser.add_argument('--json_path', default = '/data1/sap/MessyTable/labels/val.json')
    parser.add_argument('--json_path', default = '/data1/sap/MessyTable/labels/test.json')
    parser.add_argument('--save_dir', default= '/data3/sap/VEDet/data/Messytable/')
    parser.add_argument('--img_dir', default = '/data1/sap/MessyTable/images')
    parser.add_argument('--type', type=str, default = 'test')

    args = parser.parse_args()

    json = json_maker([], args.json_path, 0)
    json.load()

    class_names = list(CLASS.keys())
    class_ids = list(CLASS.values())

    all_scene_ids = json.get_all_scenes() 

    print('scene: {}'.format(len(all_scene_ids)))
    infos = fill_infos(all_scene_ids, json)
    print('sample: {}'.format(len(infos)))

    metadata = dict(version='v1.0-trainval')
    data = dict(infos=infos, metadata=metadata)
    info_path = os.path.join(args.save_dir, 'messytable_infos_{}.pkl'.format(args.type))

    with open(info_path, 'wb') as f:
        pickle.dump(data, f)



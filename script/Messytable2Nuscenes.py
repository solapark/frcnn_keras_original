import argparse 
from scipy.spatial.transform import Rotation as R
import secrets
import numpy as np
import os

from json_maker import json_maker 

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
        info = {
            'lidar_path': scene_id,
            'token': secrets.token_hex(nbytes=16),
            'cams': dict(),
            'gt_boxes' : list(instance_summary.keys()), #inst ids
            'gt_names' : list(instance_summary.values()), #inst classes
            'cam_instances' : dict(),
            'cam_instances_valid_flags' : dict()
        }

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
                'data_path' : os.path.join(args.img_dir, path_name), 
                'type' : cam_id, 
                'intrinsic' : K,
                'extrinsic' : E, 
                'lidar2cam' : world2img, 
            }

            info['cams'].update({cam_id: cam_info})

            info['cam_instances_valid_flags'][cam_id] = {id : False for id in instance_summary.keys()}

            info['cam_instances'][cam_id] = dict()
            for instance in instances :
                inst_id = instance['inst_id']
                info['cam_instances'][cam_id][inst_id] = {'id': inst_id, 'cls' : instance['subcls'], 'bbox' : instance['pos']}
                info['cam_instances_valid_flags'][cam_id][inst_id] = True


        print(info)

        infos.append(info)

    return infos

if __name__ == '__main__' :
    parser=argparse.ArgumentParser()
    parser.add_argument('--json_path', default = '/data1/sap/MessyTable/labels/test.json')
    parser.add_argument('--save_dir', default= '/data1/sap/MessyTable/labels/')
    parser.add_argument('--img_dir', default = '/data1/sap/MessyTable/images')
    parser.add_argument('--type', type=str, default = 'test')

    args = parser.parse_args()

    json = json_maker([], args.json_path, 0)
    json.load()

    all_scene_ids = json.get_all_scenes() 

    print('scene: {}'.format(len(all_scene_ids)))
    infos = fill_infos(all_scene_ids, json)
    print('sample: {}'.format(len(infos)))

    metadata = dict(version='v1.0-trainval')
    data = dict(infos=infos, metadata=metadata)
    info_path = osp.join(args.save_dir, 'messytable_infos_{}.pkl'.format(args.type))

    with open(info_path, 'wb') as f:
        pickle.dump(data, f)



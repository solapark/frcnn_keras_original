import argparse 
from scipy.spatial.transform import Rotation as R
import secrets
import numpy as np
import os
import pickle
import cv2

from json_maker import json_maker 

CLASS = {'water1': 1, 'water2': 2, 'pepsi': 3, 'coca1': 4, 'coca2': 5, 'coca3': 6, 'coca4': 7, 'tea1': 8, 'tea2': 9, 'yogurt': 10, 'ramen1': 11, 'ramen2': 12, 'ramen3': 13, 'ramen4': 14, 'ramen5': 15, 'ramen6': 16, 'ramen7': 17, 'juice1': 18, 'juice2': 19, 'can1': 20, 'can2': 21, 'can3': 22, 'can4': 23, 'can5': 24, 'can6': 25, 'can7': 26, 'can8': 27, 'can9': 28, 'ham1': 29, 'ham2': 30, 'pack1': 31, 'pack2': 32, 'pack3': 33, 'pack4': 34, 'pack5': 35, 'pack6': 36, 'snack1': 37, 'snack2': 38, 'snack3': 39, 'snack4': 40, 'snack5': 41, 'snack6': 42, 'snack7': 43, 'snack8': 44, 'snack9': 45, 'snack10': 46, 'snack11': 47, 'snack12': 48, 'snack13': 49, 'snack14': 50, 'snack15': 51, 'snack16': 52, 'snack17': 53, 'snack18': 54, 'snack19': 55, 'snack20': 56, 'snack21': 57, 'snack22': 58, 'snack23': 59, 'snack24': 60, 'green_apple': 61, 'red_apple': 62, 'tangerine': 63, 'lime': 64, 'lemon': 65, 'yellow_quince': 66, 'green_quince': 67, 'white_quince': 68, 'fruit1': 69, 'fruit2': 70, 'peach': 71, 'banana': 72, 'fruit3': 73, 'pineapple': 74, 'fruit4': 75, 'strawberry': 76, 'cherry': 77, 'red_pimento': 78, 'green_pimento': 79, 'carrot': 80, 'cabbage1': 81, 'cabbage2': 82, 'eggplant': 83, 'bread': 84, 'baguette': 85, 'sandwich': 86, 'hamburger': 87, 'hotdog': 88, 'donuts': 89, 'cake': 90, 'onion': 91, 'marshmallow': 92, 'mooncake': 93, 'shirimpsushi': 94, 'sushi1': 95, 'sushi2': 96, 'big_spoon': 97, 'small_spoon': 98, 'fork': 99, 'knife': 100, 'big_plate': 101, 'small_plate': 102, 'bowl': 103, 'white_ricebowl': 104, 'blue_ricebowl': 105, 'black_ricebowl': 106, 'green_ricebowl': 107, 'black_mug': 108, 'gray_mug': 109, 'pink_mug': 110, 'green_mug': 111, 'blue_mug': 112, 'blue_cup': 113, 'orange_cup': 114, 'yellow_cup': 115, 'big_wineglass': 116, 'small_wineglass': 117, 'glass1': 118, 'glass2': 119, 'glass3': 120, 'bg':121}

# Define the draw function
def draw(image_path, cxcywh, est_cxcy):
    # Load the image
    img = cv2.imread(image_path)

    # Iterate over instances and draw bounding boxes and estimated points
    for i in range(len(cxcywh)):
        x, y, w, h = cxcywh[i]
        est_x, est_y = est_cxcy[i]

        # Draw bounding box
        cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

        # Mark estimated point
        cv2.circle(img, (int(est_x), int(est_y)), 5, (0, 0, 255), -1)

    return img

def mv_DLT(Ps, pnts):
    # Ps #(num_cams, 4, 4)
    # pnts #(num_cams, 2) 

    pnts = pnts.reshape((-1, 2, 1))
    first_row = pnts[:, 1] * Ps[:, 2] - Ps[:, 1]
    second_row = Ps[:, 0] - pnts[:, 0] * Ps[:, 2]

    A = np.concatenate([first_row, second_row], 0).reshape((-1, 4))
    #print(pnts)
    #print('P1: ')
    #print(Ps[0])
    #print('P2: ')
    #print(Ps[1])
    #print('A: ')
    #print(A)
 
    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)
 
    #print('Triangulated point: ')
    #print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]

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
            'pickle_path': os.path.join(args.pickle_dir, '%s.pickle'%(scene_id)),
            'cams': dict(),
            'valid_flags': np.ones((num_inst,)),
            'gt_ids' : inst_ids, 
            'gt_names' : None, #inst classes
            'cam_instances' : dict(),
            'cam_instances_valid_flags' : dict(),
            'inst_3dp' : None, #(num_inst, 3) #cx,cy,cz
            'inst_proj_2dp' : None, #(num_inst, num_cam, 2) #cx,cy
            'pred_box_idx' : None, #(num_inst, num_cam) #cam1_idx,cam2_idx,cam3_idx
            'probs' : None #(num_inst, num_cam) 
        }

        gt_names = [class_names[class_ids.index(cls)] for cls in instance_summary.values()]
        info.update(gt_names = gt_names)

        scene = json.get_scene(scene_id)
        all_cam_idx = json.get_all_cams(scene) 

        cam_instances_list = []
        cam_instances_valid_flags_list = []
        pred_box_idx_list = []
        prob_list = []
        for cam_id in all_cam_idx:
            intrinsic = intrinsics[cam_id]
            K = np.eye(4)
            K[:3, :3] = np.array(intrinsic).reshape(3, 3)

            extrinsic = json.get_extrinsics(scene_id, cam_id)
            E = np.eye(4)
            E[0:3, 0:3] = R.from_euler('xyz', extrinsic[3:]).as_matrix()
            E[0:3, 3] = np.array(extrinsic[:3])

            world2img = K @ np.linalg.inv(E)

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

            cam_instances = np.zeros((num_inst, 4))
            cam_instances_valid_flags = np.zeros((num_inst, ))
            pred_box_idx = -np.ones((num_inst, ))
            probs = -np.ones((num_inst, ))

            for instance in instances :
                inst_id = instance['inst_id']
                inst_idx = inst_ids.index(inst_id)
                x1, y1, x2, y2 = instance['pos']
                cx, cy, w, h = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
                cam_instances[inst_idx] = cx, cy, w, h
                cam_instances_valid_flags[inst_idx] = 1
                pred_box_idx[inst_idx] = int(instance['pred_id'])
                probs[inst_idx] = instance['prob']

            cam_instances_list.append(cam_instances)
            cam_instances_valid_flags_list.append(cam_instances_valid_flags)
            pred_box_idx_list.append(pred_box_idx)
            prob_list.append(probs)

        info.update(cam_instances = np.stack(cam_instances_list, axis=0))
        info.update(cam_instances_valid_flags = np.stack(cam_instances_valid_flags_list, axis=0))
        info.update(pred_box_idx = np.stack(pred_box_idx_list, axis=0).transpose(1,0))
        info.update(probs = np.stack(prob_list, axis=0))
        #print(info)

        inst_3dp_list = []
        inst_2dp_list = []
        for inst_id in info['gt_ids'] :
            inst_idx = inst_ids.index(inst_id)
            valid_cam_ids = [ str(cam_idx+1) for cam_idx, cam_valid in enumerate(info["cam_instances_valid_flags"]) if cam_valid[inst_idx]==1]

            Ps = []
            pnts = []
            for cam_id in valid_cam_ids :
            #for cam_id in list(['1','2','3']) :
                if len(valid_cam_ids) < 1 :
                    print('len(valid_cam_ids) < 2 ')
                P = info['cams'][cam_id]['world2img']
                cam_idx = int(cam_id) - 1
                inst_pos = info["cam_instances"][cam_idx][inst_idx]
                inst_cxcy = inst_pos[0], inst_pos[1]

                Ps.append(P)
                pnts.append(inst_cxcy)

            Ps = np.stack(Ps, 0)
            pnts = np.stack(pnts, 0)
            
            inst_3dp = mv_DLT(Ps, pnts)
            homo_3dp = np.concatenate([inst_3dp, [1]]).reshape((4, 1))
            
            inst_2dp_list_in_cams = []
            #for cam_id in valid_cam_ids :
            for cam_id in all_cam_idx :
                if cam_id in valid_cam_ids :
                    P = info['cams'][cam_id]['world2img']
                    cam_2dp = P @ homo_3dp
                    cam_2dp = (cam_2dp[:2]/cam_2dp[2]).reshape(2, )
                    inst_2dp_list_in_cams.append(cam_2dp)
                else :
                    inst_2dp_list_in_cams.append(np.zeros((2,)))


            inst_3dp_list.append(homo_3dp.reshape(-1,)[:3])
            inst_2dp_list.append(np.array(inst_2dp_list_in_cams))

        info.update(inst_3dp = np.stack(inst_3dp_list, axis=0))
        info.update(inst_proj_2dp = np.stack(inst_2dp_list, axis=0))


        if args.debug :
            cxcywh = info['cam_instances'] #(num_cam, num_inst, 4)
            est_cxcy = info['inst_proj_2dp'].transpose(1,0,2) #(num_cam, num_inst, 4)

            # List of image paths for all cameras
            image_paths = [info['cams'][cam_id]['img_path'] for cam_id in all_cam_idx]

            # Load all images, draw on them, and store them in a list
            drawn_images = []
            for i in range(len(all_cam_idx)):
                img = draw(image_paths[i], cxcywh[i], est_cxcy[i])

                # Resize to half size
                img = cv2.resize(img, (0, 0), fx=args.debug_img_save_ratio, fy=args.debug_img_save_ratio)  # Half the dimensions
                drawn_images.append(img)

            # Get the dimensions of the first image
            height, width, _ = drawn_images[0].shape

            # Create a blank canvas to stack the images vertically
            stacked_image = np.zeros((height * len(drawn_images), width, 3), dtype=np.uint8)

            # Iterate through the images and stack them vertically
            for i, img in enumerate(drawn_images):
                stacked_image[i * height:(i + 1) * height, :] = img

            # Save the stacked image
            os.makedirs(args.debug_img_save_dir, exist_ok=True)
            output_path = os.path.join(args.debug_img_save_dir, '%s.png'%(scene_id))
            cv2.imwrite(output_path, stacked_image)

            print("Stacked image saved at:", output_path)


        infos.append(info)

    return infos

if __name__ == '__main__' :
    parser=argparse.ArgumentParser()
    #parser.add_argument('--json_path', default = '/data1/sap/MessyTable/labels/train.json')
    #parser.add_argument('--json_path', default = '/data1/sap/MessyTable/labels/val.json')
    #parser.add_argument('--json_path', default = '/data1/sap/MessyTable/labels/test.json')
    parser.add_argument('--json_path')
    #parser.add_argument('--save_dir', default= '/data3/sap/VEDet/data/Messytable/')
    parser.add_argument('--save_dir')
    parser.add_argument('--img_dir', default = '/data1/sap/MessyTable/images')
    #parser.add_argument('--type', type=str, default = 'train')
    #parser.add_argument('--type', type=str, default = 'val')
    #parser.add_argument('--type', type=str, default = 'test')
    parser.add_argument('--type', type=str)
    #parser.add_argument('--pickle_dir', type=str, default = '/data3/sap/frcnn_keras_original/pickle/messytable/mvdet/reid_input/train')
    parser.add_argument('--pickle_dir', type=str)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--debug_img_save_dir", default = '/data3/sap/frcnn_keras_original/DLT_debug')
    parser.add_argument("--debug_img_save_ratio", default = .3)

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

    os.makedirs(args.save_dir, exist_ok=True)
    with open(info_path, 'wb') as f:
        pickle.dump(data, f)



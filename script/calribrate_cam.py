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

import numpy as np

def Normalization(nd, x):
    '''
    Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3).

    Input
    -----
    nd: number of dimensions, 3 here
    x: the data to be normalized (directions at different columns and points at rows)
    Output
    ------
    Tr: the transformation matrix (translation plus scaling)
    x: the transformed data
    '''

    x = np.asarray(x)
    m, s = np.mean(x, 0), np.std(x)
    if nd == 2:
        Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        Tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])
        
    Tr = np.linalg.inv(Tr)
    x = np.dot( Tr, np.concatenate( (x.T, np.ones((1,x.shape[0]))) ) )
    x = x[0:nd, :].T

    return Tr, x


def DLTcalib(nd, xyz, uv):
    '''
    Camera calibration by DLT using known object points and their image points.

    Input
    -----
    nd: dimensions of the object space, 3 here.
    xyz: coordinates in the object 3D space.
    uv: coordinates in the image 2D space.

    The coordinates (x,y,z and u,v) are given as columns and the different points as rows.

    There must be at least 6 calibration points for the 3D DLT.

    Output
    ------
     L: array of 11 parameters of the calibration matrix.
     err: error of the DLT (mean residual of the DLT transformation in units of camera coordinates).
    '''
    if (nd != 3):
        raise ValueError('%dD DLT unsupported.' %(nd))
    
    # Converting all variables to numpy array
    xyz = np.asarray(xyz)
    uv = np.asarray(uv)

    n = xyz.shape[0]

    # Validating the parameters:
    if uv.shape[0] != n:
        raise ValueError('Object (%d points) and image (%d points) have different number of points.' %(n, uv.shape[0]))

    if (xyz.shape[1] != 3):
        raise ValueError('Incorrect number of coordinates (%d) for %dD DLT (it should be %d).' %(xyz.shape[1],nd,nd))

    if (n < 6):
        raise ValueError('%dD DLT requires at least %d calibration points. Only %d points were entered.' %(nd, 2*nd, n))
        
    # Normalize the data to improve the DLT quality (DLT is dependent of the system of coordinates).
    # This is relevant when there is a considerable perspective distortion.
    # Normalization: mean position at origin and mean distance equals to 1 at each direction.
    Txyz, xyzn = Normalization(nd, xyz)
    Tuv, uvn = Normalization(2, uv)

    A = []

    for i in range(n):
        x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
        u, v = uvn[i, 0], uvn[i, 1]
        A.append( [x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u] )
        A.append( [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v] )

    # Convert A to array
    A = np.asarray(A) 

    # Find the 11 parameters:
    U, S, V = np.linalg.svd(A)

    # The parameters are in the last line of Vh and normalize them
    L = V[-1, :] / V[-1, -1]
    #print(L)
    # Camera projection matrix
    H = L.reshape(3, nd + 1)
    #print(H)

    # Denormalization
    # pinv: Moore-Penrose pseudo-inverse of a matrix, generalized inverse of a matrix using its SVD
    H = np.dot( np.dot( np.linalg.pinv(Tuv), H ), Txyz )
    #print(H)
    H = H / H[-1, -1]
    #print(H)
    L = H.flatten()
    #print(L)

    # Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates):
    uv2 = np.dot( H, np.concatenate( (xyz.T, np.ones((1, xyz.shape[0]))) ) ) 
    uv2 = uv2 / uv2[2, :] 
    # Mean distance:
    err = np.sqrt( np.mean(np.sum( (uv2[0:2, :].T - uv)**2, 1)) ) 

    return L, err

def DLT(xyz, uv):
#def DLT():
    # Known 3D coordinates
    #xyz = [[-875, 0, 9.755], [442, 0, 9.755], [1921, 0, 9.755], [2951, 0.5, 9.755], [-4132, 0.5, 23.618], [-876, 0, 23.618]]
    # Known pixel coordinates
    #uv = [[76, 706], [702, 706], [1440, 706], [1867, 706], [264, 523], [625, 523]]
    xyz = xyz.reshape((-1, 3))
    uv = uv.reshape((-1, 2))

    nd = 3
    P, err = DLTcalib(nd, xyz, uv)
    #print('Matrix')
    #print(P)
    #print('\nError')
    #print(err)
    return P.reshape((3, 4))

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

def fill_infos(all_scene_ids, json, target_cam_id, matrix_camera, distortion_coeffs):
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
        print(scene_id)
        instance_summary = json.get_instance_summary(scene_id)
        #num_inst = len(instance_summary) if args.rpn else args.num_inst
        num_inst = len(instance_summary) 
        inst_ids = list(instance_summary.keys())
        info = {
            'scene_id': scene_id,
            'scene_token': secrets.token_hex(nbytes=16),
            'cams': dict(),
            'valid_flags': np.ones((num_inst,)),
            'is_filled': np.zeros((num_inst,)),
            'gt_ids' : inst_ids, 
            'gt_names' : None, #inst classes
            'cam_instances' : dict(),
            'cam_instances_valid_flags' : dict(),
            'inst_3dp' : None, #(num_inst, 3) #cx,cy,cz
            'inst_proj_2dp' : None, #(num_inst, num_cam, 2) #cx,cy
            'pred_box_idx_org' : None, #(num_inst, num_cam) #cam1_idx,cam2_idx,cam3_idx
            'pred_box_idx' : None, #(num_inst, num_cam) #cam1_idx,cam2_idx,cam3_idx
            'probs' : None #(num_inst, num_cam) 
        }

        info['is_filled'][:len(inst_ids)] = 1

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
                pred_box_idx[inst_idx] = int(instance['pred_id']) if args.rpn else int(inst_id)-1
                probs[inst_idx] = instance['prob'] if args.rpn else 1.

            cam_instances_list.append(cam_instances)
            cam_instances_valid_flags_list.append(cam_instances_valid_flags)
            pred_box_idx_list.append(pred_box_idx)
            prob_list.append(probs)

        info.update(cam_instances = np.stack(cam_instances_list, axis=0))
        info.update(cam_instances_valid_flags = np.stack(cam_instances_valid_flags_list, axis=0))
        info.update(pred_box_idx = np.stack(pred_box_idx_list, axis=0).transpose(1,0))
        info.update(pred_box_idx_org = np.stack(pred_box_idx_list, axis=0).transpose(1,0))
        info.update(probs = np.stack(prob_list, axis=0))
        #print(info)

        inst_3dp_list = []
        pnts_target = []
        for inst_id in info['gt_ids'] :
            inst_idx = inst_ids.index(inst_id)
            valid_cam_ids = [ str(cam_idx+1) for cam_idx, cam_valid in enumerate(info["cam_instances_valid_flags"]) if cam_valid[inst_idx]==1]

            Ps = []
            pnts = []
            if target_cam_id not in valid_cam_ids :
                continue

            if len(valid_cam_ids) < 3 :
                continue

            for cam_id in valid_cam_ids :
                cam_idx = int(cam_id) - 1
                inst_pos = info["cam_instances"][cam_idx][inst_idx]
                inst_cxcy = inst_pos[0], inst_pos[1]

                if cam_id == target_cam_id :
                    pnts_target.append(inst_cxcy)   
                    continue

                P = info['cams'][cam_id]['world2img']

                Ps.append(P)
                pnts.append(inst_cxcy)

            Ps = np.stack(Ps, 0)
            pnts = np.stack(pnts, 0)
            
            inst_3dp = mv_DLT(Ps, pnts)
            #print(inst_3dp)
            inst_3dp_list.append(inst_3dp)

        inst_3dp_list = np.array(inst_3dp_list)
        pnts_target = np.array(pnts_target)

        H = DLT(inst_3dp_list, pnts_target) #(3, 4)
        world2img = np.concatenate([H, [[0., 0., 0., 1.]]], 0)
        intrinsic = intrinsics[target_cam_id]
        K = np.eye(4)
        K[:3, :3] = np.array(intrinsic).reshape(3, 3)
        E = np.linalg.inv (np.linalg.inv(K) @ world2img) 
        #world2img = K @ np.linalg.inv(E)

        vector_translation = E[0:3, 3]
        vector_rotation = R.from_matrix(E[0:3, 0:3]).as_euler('xyz', degrees=False)
        extrinsics = np.concatenate([vector_translation,vector_rotation]).squeeze().tolist()

        json.insert_extrinsics(scene_id, target_cam_id, extrinsics)

            #retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera( [inst_3dp_list.astype('float32')], [pnts_target.astype('float32')], (1920, 1080), matrix_camera, distortion_coeffs, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        
        image_path = info['cams']['1']['img_path']
        img = cv2.imread(image_path)
        #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(inst_3dp_list, pnts_target, img.shape[::-1][:2],None,None)
        success, vector_rotation, vector_translation = cv2.solvePnP(inst_3dp_list, pnts_target, matrix_camera, distortion_coeffs, flags=0)
        nose_end_point2D, jacobian = cv2.projectPoints(inst_3dp_list, vector_rotation, vector_translation, matrix_camera, distortion_coeffs,)
        '''
        extrinsics = np.concatenate([vector_translation,vector_rotation]).squeeze().tolist()
        json.insert_extrinsics(scene_id, target_cam_id, extrinsics)

        nose_end_point2D, jacobian = cv2.projectPoints(inst_3dp_list, vector_rotation, vector_translation, matrix_camera, distortion_coeffs,)

        intrinsic = intrinsics[target_cam_id]
        K = np.eye(4)
        K[:3, :3] = np.array(intrinsic).reshape(3, 3)

        extrinsic = json.get_extrinsics(scene_id, target_cam_id)
        E = np.eye(4)
        #E[0:3, 0:3] = R.from_euler('xyz', extrinsic[3:]).as_matrix()
        #E[0:3, 0:3] = cv2.Rodrigues(vector_rotation)[0]
        E[0:3, 0:3] = cv2.Rodrigues(rvecs[0])[0]
        #E[0:3, 3] = np.array(extrinsic[:3])
        #E[0:3, 3] = np.array(vector_translation).squeeze()
        E[0:3, 3] = np.array(tvecs[0]).squeeze()
        '''
        extrinsic = json.get_extrinsics(scene_id, target_cam_id)
        E = np.eye(4)
        E[0:3, 0:3] = R.from_euler('xyz', extrinsic[3:]).as_matrix()
        E[0:3, 3] = np.array(extrinsic[:3])
        world2img = K @ np.linalg.inv(E)

        homo_3dp_list = np.concatenate([inst_3dp_list, np.ones((len(inst_3dp_list), 1))], -1).T #(4, num_gt)

        cam_2dp = (world2img @ homo_3dp_list).T #(num_gt, 4)
        cam_2dp = (cam_2dp[:, :2]/cam_2dp[:, 2:3]) #(num_gt, 2)

        print('inst_3dp', inst_3dp_list)
        print('inst_2dp', pnts_target)
        print('proj_2dp', cam_2dp)
        print('err', np.absolute(cam_2dp-pnts_target))
        print('nose_end_point2D', nose_end_point2D[0])

        #world2img = K @ np.linalg.inv(E)
        #homo_3dp = np.concatenate([inst_3dp_list[0], [1]]).reshape((4, ))

        #cam_2dp = world2img @ homo_3dp
        #cam_2dp = (cam_2dp[:2]/cam_2dp[2]).reshape(2, )
       
        #pnt2d = H @ homo_3dp 
        #pnt2d = pnt2d[:2]/pnt2d[2]
        #print('pnt2d', pnt2d)

        infos.append(info)

    return infos

if __name__ == '__main__' :
    parser=argparse.ArgumentParser()
    parser.add_argument('--src_json_path')
    parser.add_argument('--dst_json_path')
    parser.add_argument('--target_cam_id', type=str)
    parser.add_argument("--rpn", action="store_true", default=False)
    parser.add_argument("--num_inst", default=100)
    parser.add_argument('--img_dir', default = '/data1/sap/MessyTable/images')
    args = parser.parse_args()

    #DLT()
    np.set_printoptions(suppress=True)

    dst_json = json_maker([], args.src_json_path, 0)
    dst_json.load()
    dst_json.path = args.dst_json_path
    
    class_names = list(CLASS.keys())
    class_ids = list(CLASS.values())

    all_scene_ids = dst_json.get_all_scenes() 
    intrinsic = dst_json.get_intrinsics()[args.target_cam_id]
    matrix_camera = np.array(intrinsic, dtype='double').reshape((3,3))
    distortion_coeffs = np.zeros((4, 1))

    fill_infos(all_scene_ids, dst_json, args.target_cam_id, matrix_camera, distortion_coeffs)

    dst_json.save()

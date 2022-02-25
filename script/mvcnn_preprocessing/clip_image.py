from tqdm import tqdm
import argparse 
import cv2
import os, sys
import argparse 

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from json_maker import json_maker

if __name__ == '__main__' :
    parser=argparse.ArgumentParser()
    parser.add_argument('--json_path', default = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+asnet.json', help = 'Path to decide how to crop')
    parser.add_argument('--save_dir', default= '/data3/sap/mvcnn/svdet+asnet',help = 'Path(parent directory) to images for saving')

    parser.add_argument('--image_dir', default = '/data1/sap/MessyTable/images', help = 'Path to images for classification')
    parser.add_argument('--num_cam', type=int, default = 3)

    args = parser.parse_args()

    json = json_maker([], args.json_path, args.num_cam)
    json.load()
    all_scene_nums = json.get_all_scenes() 

    os.makedirs(args.save_dir, exist_ok=True)
    for scene_num in tqdm(all_scene_nums) :
        scene = json.get_scene(scene_num)
        cam_ids = json.get_all_cams(scene) 
 
        for cam_id in cam_ids :
            cam = scene['cameras'][cam_id]
            img_name, instances = json.get_all_inst(cam)

            img = cv2.imread(os.path.join(args.image_dir, img_name))

            for inst in instances:
                inst_id = inst['inst_id']
                subcls = inst['subcls']
                x1, y1, x2, y2 = map(int, inst['pos'])

                save_name = '%s-%d-%d_%d.jpg' %(scene_num, int(cam_id), int(inst_id), int(subcls))
                save_path = os.path.join(args.save_dir, save_name)

                cropped_img = img[y1: y2, x1: x2]
                cropped_img = cv2.resize(cropped_img,(256,256))

                cv2.imwrite(save_path, cropped_img)

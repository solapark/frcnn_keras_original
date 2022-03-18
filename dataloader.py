import json
import numpy as np
import os

from data.image_dataloader import IMAGE_DATALOADER
#from data.label_dataloader import LABEL_DATALOADER
from data.base_dataloader import BASE_DATALOADER
from data.extrins_dataloader import EXTRINS_DATALOADER
from data.pickle_dataloader import PICKLE_DATALOADER

class DATALOADER :
    #def __init__(self, json_path, dataset_name, im_size, mode, batch_size, shuffle):
    def __init__(self, args, mode, path):
        self.args = args
        self.batch_size = args.batch_size
        self.num_cam = args.num_cam
        self.num_valid_cam = args.num_valid_cam
        self.num_cls = args.num_cls
        self.num2cls = args.num2cls
        self.cls2num = args.cls2num
        self.base_dir = args.base_dir
        self.dataset = args.dataset
        self.mode = mode
        self.img_dir = args.messytable_img_dir if args.dataset == 'MESSYTABLE' else None
        self.demo_file = args.demo_file
        self.width, self.height = args.width, args.height
        self.resized_width, self.resized_height = (args.resized_width, args.resized_height) if (self.mode != 'val') else (self.width, self.height)

        self.shuffle = True if(self.mode == 'train') else False
        #self.shuffle = False

        dataset_file = self.demo_file if self.mode == 'demo' else '%s.json' %(self.mode) 
        dataset_path = path 

        data = self.get_data_from_json(dataset_path)
        self.img_path_list = self.get_img_path_list(data) #X

        self.image_dataloader = IMAGE_DATALOADER(self.img_path_list, self.batch_size, self.resized_width, self.resized_height, self.num_valid_cam)

        if self.mode == 'train' or self.mode == 'val' or self.mode == 'val_models' or self.mode == 'draw_json':
            self.resized_instance_list = self.get_instance_list(dataset_path, data, self.width, self.height, self.resized_width, self.resized_height)#Y
            self.label_dataloader = BASE_DATALOADER(self.resized_instance_list, self.batch_size)

        if self.mode == 'save_rpn_feature' or self.mode == 'save_ven_feature' or self.mode == 'demo' or self.mode == 'draw_json':
            self.image_path_dataloader = BASE_DATALOADER(self.img_path_list, self.batch_size)

        if self.args.is_use_epipolar : 
            extrins_list = self.get_extrins_list(data)
            self.extrins_dataloader = EXTRINS_DATALOADER(extrins_list, self.batch_size)

        if self.args.freeze_rpn :
            self.rpn_dataloader = PICKLE_DATALOADER(args, self.img_path_list, args.rpn_pickle_dir)

        if self.args.freeze_ven :
            self.ven_dataloader = PICKLE_DATALOADER(args, self.img_path_list, args.ven_pickle_dir)

        self.indices = np.arange(len(self.image_dataloader))
        self.on_epoch_end()

    def get_data_from_json(self, input_path):
        with open(input_path) as fp: return json.load(fp)

    def get_img_path_list(self, json):
        img_path_list = []
        for scene_content in list(json['scenes'].values()) :
            path_dict = dict()

            for cam_idx, camera_content in list(scene_content['cameras'].items()) :
                if int(cam_idx) > self.num_valid_cam : continue
                path_dict[int(cam_idx)] = os.path.join(self.img_dir, camera_content['pathname'])
            img_path_list.append(path_dict)
        return img_path_list


    def get_extrins_list(self, json):
        extrins_list_of_list = [] 
        for scene_content in list(json['scenes'].values()):
            extrins_list = []
            for cam_num, camera_content in scene_content['cameras'].items():
                cam_num = int(cam_num)
                if self.args.dataset == 'MESSYTABLE' : cam_num -= 1
                if cam_num >= self.args.num_valid_cam : break
                extrins_list.append(camera_content['extrinsics'])
            extrins_list_of_list.append(extrins_list)
        return extrins_list_of_list

    def get_instance_list(self, dataset_path, json, width, height, resized_width, resized_height):
        filename = dataset_path.split('/')[-1]
        is_class_minus_one = (self.args.dataset == 'MESSYTABLE' and filename in ['train.json', 'val.json', 'test.json', 'test_easy.json', 'test_medium.json', 'test_hard.json', 'gt+asnet+majority+nms.json', 'gt+triplenet+majority+nms.json', 'gt+majority.json']) 
        #is_class_minus_one = (self.args.dataset == 'MESSYTABLE' and filename in ['train.json', 'val.json', 'test.json', 'tmp.json']) 
        zoom_in_w = resized_width / float(width)
        zoom_in_h = resized_height / float(height)
        resized_instance_list = []
        for scene_content in list(json['scenes'].values()):
            resized_instance_dict = dict()
            for instance_num, cls in list(scene_content['instance_summary'].items()) :
                if is_class_minus_one: cls -= 1
                resized_instance_dict[instance_num] = {'cls':cls, 'instance_num':instance_num, 'resized_box':{}, 'prob':{}}
                for cam_num, camera_content in list(scene_content['cameras'].items()) :
                    cam_num = int(cam_num)
                    if self.args.dataset == 'MESSYTABLE' : cam_num -= 1
                    if cam_num >= self.args.num_valid_cam : continue
                    if instance_num in camera_content['instances'] :
                        x1, y1, x2, y2 = camera_content['instances'][instance_num]['pos']
                        x1 *= zoom_in_w                
                        x2 *= zoom_in_w                
                        y1 *= zoom_in_h                
                        y2 *= zoom_in_h                
                        resized_instance_dict[instance_num]['resized_box'][cam_num] = list(map(float, [x1, y1, x2, y2]))
                        prob = camera_content['instances'][instance_num]['prob'] if 'prob' in camera_content['instances'][instance_num] else 1
                        resized_instance_dict[instance_num]['prob'][cam_num] = prob
                if len(resized_instance_dict[instance_num]['resized_box']) == 0 :
                    resized_instance_dict.pop(instance_num)

            resized_instance_list.append(list(resized_instance_dict.values()))
        self.zoom_in_w = zoom_in_w
        self.zoom_in_h = zoom_in_h
        return resized_instance_list

    def __len__(self):
        return len(self.image_dataloader)

    def __getitem__(self, idx):
        images, labels, image_paths, extrins, rpn_results, ven_results = None,None,None,None,None, None

        images = self.image_dataloader[idx]

        if self.mode == 'train' or self.mode == 'val' or self.mode == 'val_models' or self.mode == 'draw_json':
            labels = self.label_dataloader[idx]

        if self.mode == 'save_rpn_feature' or self.mode == 'save_ven_feature' or self.mode == 'demo' or self.mode == 'draw_json':
            image_paths = self.image_path_dataloader[idx]

        if self.args.is_use_epipolar : 
            extrins = self.extrins_dataloader[idx]

        if self.args.freeze_rpn :
            rpn_results = self.rpn_dataloader[idx]

        if self.args.freeze_ven :
            ven_results = self.ven_dataloader[idx]

        items = [images, labels, image_paths, extrins, rpn_results, ven_results]

        return items

    def on_epoch_end(self):
        if(self.shuffle) : 
            np.random.shuffle(self.indices)
        self.image_dataloader.set_indices(self.indices)

        if self.mode == 'train' or self.mode == 'val' or self.mode == 'val_models'or self.mode == 'draw_json':
            self.label_dataloader.set_indices(self.indices)

        if self.mode == 'save_rpn_feature' or self.mode == 'save_ven_feature' or self.mode == 'demo' or self.mode == 'draw_json':
            self.image_path_dataloader.set_indices(self.indices)

        if self.args.is_use_epipolar : 
            self.extrins_dataloader.set_indices(self.indices)

        if self.args.freeze_rpn :
            self.rpn_dataloader.set_indices(self.indices)

        if self.args.freeze_ven :
            self.ven_dataloader.set_indices(self.indices)

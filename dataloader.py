import json
import numpy as np
import os

from data.image_dataloader import IMAGE_DATALOADER
from data.label_dataloader import LABEL_DATALOADER
from data.extrins_dataloader import EXTRINS_DATALOADER

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
        self.img_dir = args.messytable_img_dir if args.dataset == 'MESSYTABLE' else None
        self.demo_file = args.demo_file
        self.width, self.height = args.width, args.height
        self.resized_width, self.resized_height = args.resized_width, args.resized_height

        self.mode = mode
        self.shuffle = True if(self.mode == 'train') else False
        #self.shuffle = False

        dataset_file = self.demo_file if self.mode == 'demo' else '%s.json' %(self.mode) 
        dataset_path = path 

        data = self.get_data_from_json(dataset_path)
        self.img_path_list = self.get_img_path_list(data) #X

        self.image_dataloader = IMAGE_DATALOADER(self.img_path_list, self.batch_size, self.resized_width, self.resized_height, self.num_valid_cam)

        if(self.mode == 'train' or self.mode == 'val' or self.mode == 'test'):
            self.resized_instance_list = self.get_instance_list(data, self.width, self.height, self.resized_width, self.resized_height)#Y
            if self.args.is_use_epipolar : 
                extrins_list = self.get_extrins_list(data)
                self.extrins_dataloader = EXTRINS_DATALOADER(extrins_list, self.batch_size)
            self.label_dataloader = LABEL_DATALOADER(self.resized_instance_list, self.batch_size)

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

    def get_instance_list(self, json, width, height, resized_width, resized_height):
        zoom_in_w = resized_width / float(width)
        zoom_in_h = resized_height / float(height)
        resized_instance_list = []
        for scene_content in list(json['scenes'].values()):
            resized_instance_dict = dict()
            for instance_num, cls in list(scene_content['instance_summary'].items()) :
                if self.args.dataset == 'MESSYTABLE' : cls -= 1
                resized_instance_dict[instance_num] = {'cls':cls, 'resized_box':{}}
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

            resized_instance_list.append(list(resized_instance_dict.values()))
        self.zoom_in_w = zoom_in_w
        self.zoom_in_h = zoom_in_h
        return resized_instance_list

    def __len__(self):
        return len(self.image_dataloader)

    def __getitem__(self, idx):
        items = []
        if self.mode == 'demo':
            items.append( self.image_dataloader[idx])
        elif self.mode == 'train' or self.mode == 'val':
            items.extend( [self.image_dataloader[idx], self.label_dataloader[idx]])
        elif self.mode == 'test' :
            items.extend( [self.image_dataloader[idx], self.label_dataloader[idx], self.img_path_list[idx]])

        if self.args.is_use_epipolar : 
            items.append(self.extrins_dataloader[idx])
        return items

    def on_epoch_end(self):
        if(self.shuffle) : 
            np.random.shuffle(self.indices)
        self.image_dataloader.set_indices(self.indices)
        self.extrins_dataloader.set_indices(self.indices)
        if(self.mode == 'train' or self.mode == 'val' or self.mode == 'test'):
            self.label_dataloader.set_indices(self.indices)

        if self.args.is_use_epipolar : 
            self.extrins_dataloader.set_indices(self.indices)

if __name__ == '__main__' :
    mode = 'train'
    dl = DATALOADER(args, mode)
    import cv2
    import time
    from utils import get_concat_img
    from option import args
    for idx, batch in enumerate(dl):
        imgs_in_batch, labels_in_batch = batch
        for batch_idx, (imgs_in_one_batch, label_in_one_batch) in enumerate(zip(imgs_in_batch, labels_in_batch)):
            #for inst in label_in_one_batch : print(inst)
            num_inst = len(label_in_one_batch)
            color_list = [tuple(np.random.random(size=3)*256) for _ in range(num_inst)]
            img_list = []
            for cam_idx, img in enumerate(imgs_in_one_batch) :
                for inst_idx, inst in enumerate(label_in_one_batch) :
                    if cam_idx in inst['resized_box'] :
                        cls_num = inst['cls']
                        cls = dl.num2cls[cls_num]
                        x1, y1, x2, y2 = list(map(int, inst['resized_box'][cam_idx]))
                        color = color_list[inst_idx]
                        img = cv2.rectangle(img,(x1, y1),(x2, y2),color,3)
                        put_str = '%s%d' % (cls, inst_idx)
                        img = cv2.putText(img, put_str, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                img_list.append(img)
            name = 'batch%d' %(batch_idx)
            concat_img = get_concat_img(img_list)
            cv2.imshow(name, concat_img)
        cv2.waitKey()

   

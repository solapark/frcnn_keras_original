import json
import numpy as np
import os

from data.image_dataloader import IMAGE_DATALOADER
from data.label_dataloader import LABEL_DATALOADER

class DATALOADER :
    #def __init__(self, json_path, dataset_name, im_size, mode, batch_size, shuffle):
    def __init__(self, args, mode, path):
        self.batch_size = args.batch_size
        self.num_cam = args.num_cam
        self.num_valid_cam = args.num_valid_cam
        self.num_cls = args.num_cls
        self.num2cls = args.num2cls
        self.cls2num = args.cls2num
        self.base_dir = args.base_dir
        self.dataset = args.dataset
        self.img_dir = args.img_dir
        self.demo_file = args.demo_file
        self.width, self.height = args.width, args.height
        self.resized_width, self.resized_height = args.resized_width, args.resized_height

        self.mode = mode
        self.shuffle = True if(self.mode == 'train') else False
        #self.shuffle = False

        dataset_file = self.demo_file if self.mode == 'demo' else '%s.json' %(self.mode) 
        dataset_path = path 

        data = self.get_data_from_json(dataset_path)
        img_path_list = self.get_img_path_list(data) #X

        self.image_dataloader = IMAGE_DATALOADER(img_path_list, self.batch_size, self.resized_width, self.resized_height, self.num_valid_cam)

        extrinsic_list = self.get_extrinsic_list(data)
        self.extrinsic_dataloader = LABEL_DATALOADER(extrinsic_list, self.batch_size)

        if(self.mode == 'train' or self.mode == 'val' or self.mode == 'test'):
            self.resized_instance_list = self.get_instance_list(data, self.width, self.height, self.resized_width, self.resized_height)#Y
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

    def get_extrinsic_list(self, json)
        extrinsic_list = []
        for scene_content in list(json['scenes'].values()):
            cur_extrinsic_list = []
            for cam_idx, camera_content in list(scene_content['cameras'].items()) :
                if int(cam_idx) > self.num_valid_cam : continue
                    cur_extrinsic_list.append(camera_content['extrinsics'])
            extrinsic_list.append(cur_extrinsic_list)    
        return extrinsic_list

    def get_instance_list(self, json, width, height, resized_width, resized_height):
        zoom_in_w = resized_width / float(width)
        zoom_in_h = resized_height / float(height)
        resized_instance_list = []
        for scene_content in list(json['scenes'].values()):
            resized_instance_dict = dict()
            for instance_num, cls in list(scene_content['instance_summary'].items()) :
                resized_instance_dict[instance_num] = {'cls':str(cls), 'resized_box':{}}
                for cam_idx, camera_content in list(scene_content['cameras'].items()) :
                    if int(cam_idx) > self.num_valid_cam : continue
                    if instance_num in camera_content['instances'] :
                        x1, y1, x2, y2 = camera_content['instances'][instance_num]['pos']
                        x1 *= zoom_in_w                
                        x2 *= zoom_in_w                
                        y1 *= zoom_in_h                
                        y2 *= zoom_in_h                
                        resized_instance_dict[instance_num]['resized_box'][int(cam_idx)-1] = list(map(float, [x1, y1, x2, y2]))

            resized_instance_list.append(list(resized_instance_dict.values()))
        self.zoom_in_w = zoom_in_w
        self.zoom_in_h = zoom_in_h
        return resized_instance_list

    def __len__(self):
        return len(self.image_dataloader)

    def __getitem__(self, idx):
        if self.mode == 'demo':
            return self.image_dataloader[idx], self.extrinsic_dataloader[idx]
        elif self.mode == 'train' or self.mode == 'test' or self.mode == 'val':
            return self.image_dataloader[idx], self.extrinsic_dataloader[idx], self.label_dataloader[idx] 

    def on_epoch_end(self):
        if(self.shuffle) : 
            np.random.shuffle(self.indices)
        self.image_dataloader.set_indices(self.indices)
        if(self.mode == 'train' or self.mode == 'val' or self.mode == 'test'):
            self.label_dataloader.set_indices(self.indices)

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

   

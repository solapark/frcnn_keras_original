import math
import threading
import cv2
import numpy as np

class IMAGE_DATALOADER:
    def __init__(self, img_path_list, batch_size, resized_width, resized_height, num_cam):        
        self.data = img_path_list
        self.batch_size = batch_size
        self.resized_width, self.resized_height = resized_width, resized_height
        self.num_cam = num_cam

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, idx):
        # return  [[imgA_cam1, imgA_cam2, ...], [imgB_cam1, imgB_cam2, ...], ...]
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        paths_batch = [list(self.data[i].values()) for i in indices] #[[cam1_path, cam2_path, ...], [cam1_path, cam2_path, ...], ...]
        img_dict = {} #{0:{}, 1:{}, ...}
        for batch_idx in range(self.batch_size) :
            img_dict[batch_idx] = {key : None for key in range(self.num_cam)}
        
        threads = []
        for batch_idx, paths in enumerate(paths_batch):
            for cam_idx, path in enumerate(paths) : 
                t = threading.Thread(target=self.load_img, args=(path, batch_idx, cam_idx, img_dict))
                threads.append(t)
                t.start()
                #self.load_img(path, batch_idx, cam_idx, img_dict)
        for t in threads :
            t.join()

        
        #img_dict : {0:{'1':imgA_cam1, '2':imgA_cam2, ....}, 1:{'1':imgB_cam1, '2':imgB_cam2, ....}, ...}
            
        batch = []
        for images_in_one_batch in list(img_dict.values()):
            batch.append(list(images_in_one_batch.values()))

        batch_np = np.array(batch).transpose(1, 0, 2, 3, 4)
        batch_list = [cam_batch for cam_batch in batch_np]
        return batch_list

    def load_img(self, path, batch_idx, cam_idx, img_dict):
        img = cv2.imread(path)
        #print(path)
        img = cv2.resize(img, (self.resized_width, self.resized_height), interpolation=cv2.INTER_CUBIC)
        img_dict[batch_idx][cam_idx] = img

    def set_indices(self, indices):
        self.indices = indices

import os
from script import utils

class Simple_label :
    def __init__(self, path, class_list, img_base_path):
        self.path = path
        self.cls_list = class_list
        self.img_base_path = img_base_path

        f = open(self.path,'w')
        f.close()

    def write_txt(self, list_of_list):
        utils.list_of_list2txt(self.path, list_of_list, 'a')

    def write(self, list_of_list):
        utils.list2csv(self.path, list_of_list, 'a')

    def get_simple_label(self, img_name, instances):
        img_path = os.path.join(self.img_base_path, img_name)
        simple_label = []
        for instance in instances :
            cls_idx = instance['subcls']
            #cls_name = self.cls_list[cls_idx]
            cls_name = cls_idx
            x1, y1, x2, y2 = map(round, instance['pos'])
            if 'prob' in instance : 
                simple_label.append([img_path, x1, y1, x2, y2, cls_name, instance['prob']])
            else :
                simple_label.append([img_path, x1, y1, x2, y2, cls_name])
        return simple_label

    '''
    def change_format(self, simple_label):
        new = []
        for instance in simple_label :
            _, x1, y1, x2, y2, cls_name, prob = instance
            new.append([cls_name, prob, x1, y1, x2, y2]) 
        return new
    '''

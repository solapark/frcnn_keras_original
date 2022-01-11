import os
from script.utils import list2csv

class Simple_label :
    def __init__(self, path, class_list, img_base_path):
        self.path = path
        self.cls_list = class_list
        self.img_base_path = img_base_path

    def write(self, list_of_list):
        list2csv(self.path, list_of_list, 'a')

    def get_simple_label(self, img_name, instances):
        img_path = os.path.join(self.img_base_path, img_name)
        simple_label = []
        for instance in instances :
            cls_idx = instance['subcls']
            #cls_name = self.cls_list[cls_idx]
            cls_name = cls_idx
            x1, y1, x2, y2 = map(int, instance['pos'])
            simple_label.append([img_path, x1, y1, x2, y2, cls_name])
        return simple_label

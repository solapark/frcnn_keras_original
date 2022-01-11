import os
from utils import get_list_from_file, get_name_from_path

class Yolo_label :
    def __init__(self, img_dir, label_dir, cls_list):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_width = 640
        self.img_height = 360
        self.cls_list = cls_list

    def get_label_path (self, img_path):
        img_name = get_name_from_path(img_path)
        label_name = img_name.replace('.jpg', '.txt')
        return os.path.join(self.label_dir, label_name)

    def line2label(self, line):
        cls, cx, cy, w, h = line.split()
        cx = int(float(cx)*self.img_width)
        cy = int(float(cy)*self.img_height)
        w = int(float(w)*self.img_width)
        h = int(float(h)*self.img_height)
        x1 = cx - int(w/2)
        x2 = x1 + w
        y1 = cy -  int(h/2)
        y2 = y1 + h
        return cls, [x1, y1, x2, y2]

    def lines2labels(self, lines):
        result = []
        for line in lines :
            result.append(self.line2label(line)) 
        return result

    def get_labels(self, img_path):
        '''
        return labels
            labels : list of label
                label : [cls, [x1, y1, x2, y2]]
        '''
        label_path = self.get_label_path(img_path)
        lines = get_list_from_file(label_path)
        labels = self.lines2labels(lines)
        return labels

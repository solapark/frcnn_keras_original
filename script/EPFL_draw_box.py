import cv2
import glob
from utils import get_value_in_pattern, get_list_from_file, list2csv, sortBy

label_info_path = '/data3/sap/EPFL_MVMC/instance_label.csv'
lable_info_header = ['frame', 'cam', 'cls', 'obj', 'x1', 'y1', 'x2', 'y2', 'inst']
label_base_path = '/data3/sap/EPFL_MVMC/label/bounding_boxes_EPFL_cross/gt_files242_%s/visible_frame/*.txt'
image_src_base_dir = '/data3/sap/EPFL_MVMC/image'
image_dst_base_dir = '/data3/sap/EPFL_MVMC/image_%s'
label_path_pattern = '.*det_frame(.*)_cam(.*).txt'
cls_list = ['person', 'car', 'bus']
#cls_list = ['person']
num_cam = 6
color_list = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 255), (0, 0, 0)]

if __name__ == '__main__' :
    label_info = []
    for cls in cls_list :
        cls_label_path = label_base_path % (cls)
        image_dst_dir = image_dst_base_dir % (cls)
        label_paths = glob.glob(cls_label_path)
        for label_path in label_paths :
            frame, cam = list(map(int, get_value_in_pattern(label_path, label_path_pattern)))
            image_src_path = '%s/c%d/%08d.jpg' %(image_src_base_dir, cam, frame)
            image_dst_path = '%s/c%d/%08d.jpg' %(image_dst_dir, cam, frame)
            image = cv2.imread(image_src_path)
            coords = get_list_from_file(label_path)
            #print(label_path)
            for i, coord in enumerate(coords) :
                color = color_list[i%len(color_list)]
                coord_f = list(map(float, coord.split()))
                x1, y1, x2, y2 = list(map(int, coord_f))
                text_point = (x1-10, y1-10)
                cv2.putText(image, str(i), text_point, 1, 2, color, 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
                #cv2.imshow('image', image)
                #cv2.waitKey()
                label_info.append([frame, cam, cls, i, x1, y1, x2, y2])
                
            cv2.imwrite(image_dst_path, image)
        label_info = sortBy(label_info, [1, 0, 2])
        list2csv(label_info_path, label_info, header=lable_info_header)

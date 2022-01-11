from utils import get_file_list_from_fileform, check_pattern_exist, copy_file, split_list, csv2list, get_value_in_pattern, ls_by_pattern, list2csv
import os
from yolo_label import Yolo_label

# make mv dataset(copy) and make label for frcnn_keras

src_dir = '/data1/extra_1210_mod/images'
cam_num = [0, 1, 2]
file_names_csv = '/data1/sap/interpark_data/mv/pattern.csv'
src2dst_csv = '/data1/sap/interpark_data/mv/src2dst.csv'
dst_pattern = "%s_%d_%04d.jpg"
train_dir = '/data1/sap/interpark_data/mv/train/'
val_dir = '/data1/sap/interpark_data/mv/val/'
test_dir = '/data1/sap/interpark_data/mv/test/'
train_size = 230

all_label_path = '/data1/sap/frcnn_keras/data/%s_interpark18_%s.txt'
is_make_mv_dataset = False
is_make_label = True

def get_img_label_path(folder, file_name):
    img_file = os.path.join(folder, file_name)
    txt_file = img_file.replace('.jpg', '.txt')
    txt_file = txt_file.replace('images', 'labels')
    return img_file, txt_file

def copy_files(cls, scene_nums, cam_num, src_patterns, dst_pattern, src_dir, dst_dir, src2dst_list, dst_start_num = None):
    for scene_num in scene_nums :
        for c, pattern in zip(cam_num, src_patterns) : 
            src_final_format = pattern.replace('([0-9]*)', scene_num)
            src_final_format = src_final_format.replace('\\', '')
            src_image_file, src_txt_file = get_img_label_path(src_dir, src_final_format)

            if(dst_start_num != -1) :
                dst_final_format = dst_pattern % (cls, c, dst_start_num)
                dst_image_file, dst_txt_file = get_img_label_path(dst_dir, dst_final_format)
            else :
                dst_image_file, dst_txt_file = get_img_label_path(dst_dir, src_final_format)

            src2dst_list.append([src_image_file, dst_image_file])

            copy_file(src_image_file, dst_image_file)
            copy_file(src_txt_file, dst_txt_file)

        if(dst_start_num != -1) :
            dst_start_num += 1

def make_label(cls, scene_nums, cam_num, dst_pattern, dst_dir, dst_start_num, is_sv, all_label_file, yolo_label):
    for scene_num in scene_nums :
        for c in cam_num  : 
            dst_final_format = dst_pattern % (cls, c, dst_start_num)
            dst_image_file, dst_txt_file = get_img_label_path(dst_dir, dst_final_format)

            _, [x1, y1, x2, y2] = yolo_label.get_labels(dst_image_file)[0]
            if(is_sv):
                all_label_file.write("%s, %d, %d, %d, %d, %s\n" %(dst_image_file, x1, y1, x2, y2, cls))
            else : 
                if(c == cam_num[-1]) : 
                    all_label_file.write("%s,%d,%d,%d,%d, %s\n" %(dst_image_file, x1, y1, x2, y2, cls))
                else :
                    all_label_file.write("%s,%d,%d,%d,%d, %s," %(dst_image_file, x1, y1, x2, y2, cls))
        dst_start_num += 1



if __name__ == '__main__' : 
    if(is_make_mv_dataset):
        for folder in [train_dir, val_dir, test_dir]:
            os.system('rm -rf ' + folder)
            os.makedirs(folder)

    if(is_make_label):
        sv_train = open(all_label_path%('sv', 'train'), 'w')
        sv_val = open(all_label_path%('sv', 'val'), 'w')
        sv_test = open(all_label_path%('sv', 'test'), 'w')

        mv_train = open(all_label_path%('mv', 'train'), 'w')
        mv_val = open(all_label_path%('mv', 'val'), 'w')
        mv_test = open(all_label_path%('mv', 'test'), 'w')

        yolo_label_train = Yolo_label(train_dir, train_dir, ['dummy'])
        yolo_label_val = Yolo_label(val_dir, val_dir, ['dummy'])
        yolo_label_test = Yolo_label(test_dir, test_dir, ['dummy'])

    file_names = csv2list(file_names_csv, header=False)
    num_cls = {}
    src2dst_list = []
    cnt = -1
    for cls_and_pattern in file_names:
        cnt += 1
        #if cnt < 36 : continue 
        cls = cls_and_pattern[0].lower()
        patterns = cls_and_pattern[1:4]
        total, num_train, num_val, num_test = map(int, cls_and_pattern[4:])
        if(num_train == 0) : continue

        if cls not in num_cls :
            num_cls[cls] = 0

        scene_num_set = {cam_idx : set() for cam_idx in cam_num}
        for cam_idx, pattern in zip(cam_num, patterns) : 
            clean_pattern = pattern.replace('([0-9]*)', '[0-9]*')
            clean_pattern = clean_pattern.replace('\\', '')
            file_names = ls_by_pattern(src_dir, clean_pattern, is_full_path=False)
            for file_name in file_names : 
                scene_num = get_value_in_pattern(file_name, pattern)
                scene_num_set[cam_idx].add(scene_num)
        
        valid_scene_num = scene_num_set[cam_num[0]]
        for cam_idx in cam_num :
            valid_scene_num &= scene_num_set[cam_num[cam_idx]]

        all_scene_num = sorted(list(valid_scene_num))
        #print(all_scene_num) 

        if(len(all_scene_num) != total) : 
            print('len(all_scene_num) != total', len(all_scene_num), '!=', total)

        train_scene_num, test_scene_num = split_list(all_scene_num, num_train) 
        val_scene_num, test_scene_num = split_list(test_scene_num, num_val) 
        test_scene_num = test_scene_num[:num_test]
        for l in [train_scene_num, val_scene_num, test_scene_num] : l.sort()
        print(cnt, cls, len(train_scene_num), len(val_scene_num), len(test_scene_num), sep='\t')

        train_start_idx = num_cls[cls]
        val_start_idx = train_start_idx + len(train_scene_num)
        test_start_idx = val_start_idx + len(val_scene_num)
        num_cls[cls] = test_start_idx + len(test_scene_num)

        if(is_make_mv_dataset) :
            copy_files(cls, train_scene_num, cam_num, patterns, dst_pattern, src_dir, train_dir, src2dst_list, train_start_idx)
            copy_files(cls, val_scene_num, cam_num, patterns, dst_pattern, src_dir, val_dir, src2dst_list, val_start_idx)
            copy_files(cls, test_scene_num, cam_num, patterns, dst_pattern, src_dir, test_dir, src2dst_list, test_start_idx)

        if(is_make_label) :
            make_label(cls, train_scene_num, cam_num, dst_pattern, train_dir, train_start_idx, True, sv_train, yolo_label_train)
            make_label(cls, train_scene_num, cam_num, dst_pattern, train_dir, train_start_idx, False, mv_train, yolo_label_train)

            make_label(cls, val_scene_num, cam_num, dst_pattern, val_dir, val_start_idx, True, sv_val, yolo_label_val)
            make_label(cls, val_scene_num, cam_num, dst_pattern, val_dir, val_start_idx, False, mv_val, yolo_label_val)

            make_label(cls, test_scene_num, cam_num, dst_pattern, test_dir, test_start_idx, True, sv_test, yolo_label_test)
            make_label(cls, test_scene_num, cam_num, dst_pattern, test_dir, test_start_idx, False, mv_test, yolo_label_test)

    if(is_make_mv_dataset) : list2csv(src2dst_csv, src2dst_list, header=['src', 'dst'])
    print(num_cls)

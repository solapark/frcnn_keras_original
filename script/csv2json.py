from json_maker import json_maker 
from utils import get_list_from_file
import os
import argparse 

#class_mapping = {'water1': 53, 'water2': 79, 'pepsi': 21, 'coca1': 64, 'coca2': 75, 'coca3': 105, 'coca4': 76, 'tea1': 65, 'tea2': 1, 'yogurt': 25, 'ramen1': 13, 'ramen2': 63, 'ramen3': 85, 'ramen4': 12, 'ramen5': 26, 'ramen6': 48, 'ramen7': 69, 'juice1': 59, 'juice2': 66, 'can1': 20, 'can2': 73, 'can3': 72, 'can4': 104, 'can5': 6, 'can6': 44, 'can7': 89, 'can8': 41, 'can9': 81, 'ham1': 5, 'ham2': 55, 'pack1': 42, 'pack2': 86, 'pack3': 18, 'pack4': 102, 'pack5': 19, 'pack6': 43, 'snack1': 106, 'snack2': 91, 'snack3': 74, 'snack4': 113, 'snack5': 88, 'snack6': 78, 'snack7': 17, 'snack8': 16, 'snack9': 77, 'snack10': 87, 'snack11': 14, 'snack12': 32, 'snack13': 98, 'snack14': 35, 'snack15': 90, 'snack16': 68, 'snack17': 70, 'snack18': 67, 'snack19': 97, 'snack20': 47, 'snack21': 46, 'snack22': 71, 'snack23': 95, 'snack24': 4, 'green_apple': 52, 'red_apple': 9, 'tangerine': 49, 'lime': 50, 'lemon': 31, 'yellow_quince': 3, 'green_quince': 51, 'white_quince': 94, 'fruit1': 83, 'fruit2': 24, 'peach': 23, 'banana': 28, 'fruit3': 103, 'pineapple': 92, 'fruit4': 39, 'strawberry': 96, 'cherry': 15, 'red_pimento': 22, 'green_pimento': 99, 'carrot': 0, 'cabbage1': 45, 'cabbage2': 2, 'eggplant': 40, 'bread': 36, 'baguette': 7, 'sandwich': 37, 'hamburger': 8, 'hotdog': 10, 'donuts': 84, 'cake': 58, 'onion': 82, 'marshmallow': 80, 'mooncake': 38, 'shirimpsushi': 60, 'sushi1': 62, 'sushi2': 61, 'big_spoon': 30, 'small_spoon': 112, 'fork': 27, 'knife': 93, 'big_plate': 115, 'small_plate': 11, 'bowl': 110, 'white_ricebowl': 109, 'blue_ricebowl': 118, 'black_ricebowl': 116, 'green_ricebowl': 119, 'black_mug': 34, 'gray_mug': 100, 'pink_mug': 108, 'green_mug': 117, 'blue_mug': 101, 'blue_cup': 33, 'orange_cup': 56, 'yellow_cup': 57, 'big_wineglass': 114, 'small_wineglass': 29, 'glass1': 111, 'glass2': 54, 'glass3': 107, 'bg': 120}
class_mapping = {'water1': 0, 'water2': 1, 'pepsi': 2, 'coca1': 3, 'coca2': 4, 'coca3': 5, 'coca4': 6, 'tea1': 7, 'tea2': 8, 'yogurt': 9, 'ramen1': 10, 'ramen2': 11, 'ramen3': 12, 'ramen4': 13, 'ramen5': 14, 'ramen6': 15, 'ramen7': 16, 'juice1': 17, 'juice2': 18, 'can1': 19, 'can2': 20, 'can3': 21, 'can4': 22, 'can5': 23, 'can6': 24, 'can7': 25, 'can8': 26, 'can9': 27, 'ham1': 28, 'ham2': 29, 'pack1': 30, 'pack2': 31, 'pack3': 32, 'pack4': 33, 'pack5': 34, 'pack6': 35, 'snack1': 36, 'snack2': 37, 'snack3': 38, 'snack4': 39, 'snack5': 40, 'snack6': 41, 'snack7': 42, 'snack8': 43, 'snack9': 44, 'snack10': 45, 'snack11': 46, 'snack12': 47, 'snack13': 48, 'snack14': 49, 'snack15': 50, 'snack16': 51, 'snack17': 52, 'snack18': 53, 'snack19': 54, 'snack20': 55, 'snack21': 56, 'snack22': 57, 'snack23': 58, 'snack24': 59, 'green_apple': 60, 'red_apple': 61, 'tangerine': 62, 'lime': 63, 'lemon': 64, 'yellow_quince': 65, 'green_quince': 66, 'white_quince': 67, 'fruit1': 68, 'fruit2': 69, 'peach': 70, 'banana': 71, 'fruit3': 72, 'pineapple': 73, 'fruit4': 74, 'strawberry': 75, 'cherry': 76, 'red_pimento': 77, 'green_pimento': 78, 'carrot': 79, 'cabbage1': 80, 'cabbage2': 81, 'eggplant': 82, 'bread': 83, 'baguette': 84, 'sandwich': 85, 'hamburger': 86, 'hotdog': 87, 'donuts': 88, 'cake': 89, 'onion': 90, 'marshmallow': 91, 'mooncake': 92, 'shirimpsushi': 93, 'sushi1': 94, 'sushi2': 95, 'big_spoon': 96, 'small_spoon': 97, 'fork': 98, 'knife': 99, 'big_plate': 100, 'small_plate': 101, 'bowl': 102, 'white_ricebowl': 103, 'blue_ricebowl': 104, 'black_ricebowl': 105, 'green_ricebowl': 106, 'black_mug': 107, 'gray_mug': 108, 'pink_mug': 109, 'green_mug': 110, 'blue_mug': 111, 'blue_cup': 112, 'orange_cup': 113, 'yellow_cup': 114, 'big_wineglass': 115, 'small_wineglass': 116, 'glass1': 117, 'glass2': 118, 'glass3': 119, 'bg': 120}

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log.csv')
    parser.add_argument('--dst_path', type=str, default='/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log.json')

    args = parser.parse_args()

    jm = json_maker([], args.dst_path, 0)
    simple_label_list = get_list_from_file(args.src_path)

    for simple_label in simple_label_list :
        filepath, x1, y1, x2, y2, prob, cls_name = simple_label.split(',')
        filename_with_ext = os.path.basename(filepath)
        filename = filename_with_ext.split('.')[0]
        filename_split = filename.split('-')
        cam_num = str(int(filename_split[-1]))
        scene_num = '-'.join(filename_split[:-1])
        cls_idx = class_mapping[cls_name]

        if not jm.is_scene_in_tree(scene_num) :
            jm.insert_scene(scene_num)
        if not jm.is_cam_in_scene(scene_num, cam_num): 
            jm.insert_cam(scene_num, cam_num)
            jm.insert_path(scene_num, cam_num, filename_with_ext)

        #inst_id = jm.get_number_of_insts_in_cam(scene_num, cam_num) + 1
        inst_id = jm.get_number_of_insts_in_instance_summary(scene_num) + 1
        jm.insert_instance(scene_num, cam_num, inst_id, cls_idx, x1, y1, x2, y2, prob) 
        #if not jm.is_inst_in_instance_summary(scene_num, inst_id) : 
        #   jm.insert_instance_summary(scene_num, inst_id, 100) 
        jm.insert_instance_summary(scene_num, inst_id, cls_idx) 
    jm.sort()
    jm.save()


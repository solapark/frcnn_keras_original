from json_maker import json_maker 
from utils import get_list_from_file
import os

json_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log.json'
simple_label_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log.csv'
class_mapping = {'water1': 53, 'water2': 79, 'pepsi': 21, 'coca1': 64, 'coca2': 75, 'coca3': 105, 'coca4': 76, 'tea1': 65, 'tea2': 1, 'yogurt': 25, 'ramen1': 13, 'ramen2': 63, 'ramen3': 85, 'ramen4': 12, 'ramen5': 26, 'ramen6': 48, 'ramen7': 69, 'juice1': 59, 'juice2': 66, 'can1': 20, 'can2': 73, 'can3': 72, 'can4': 104, 'can5': 6, 'can6': 44, 'can7': 89, 'can8': 41, 'can9': 81, 'ham1': 5, 'ham2': 55, 'pack1': 42, 'pack2': 86, 'pack3': 18, 'pack4': 102, 'pack5': 19, 'pack6': 43, 'snack1': 106, 'snack2': 91, 'snack3': 74, 'snack4': 113, 'snack5': 88, 'snack6': 78, 'snack7': 17, 'snack8': 16, 'snack9': 77, 'snack10': 87, 'snack11': 14, 'snack12': 32, 'snack13': 98, 'snack14': 35, 'snack15': 90, 'snack16': 68, 'snack17': 70, 'snack18': 67, 'snack19': 97, 'snack20': 47, 'snack21': 46, 'snack22': 71, 'snack23': 95, 'snack24': 4, 'green_apple': 52, 'red_apple': 9, 'tangerine': 49, 'lime': 50, 'lemon': 31, 'yellow_quince': 3, 'green_quince': 51, 'white_quince': 94, 'fruit1': 83, 'fruit2': 24, 'peach': 23, 'banana': 28, 'fruit3': 103, 'pineapple': 92, 'fruit4': 39, 'strawberry': 96, 'cherry': 15, 'red_pimento': 22, 'green_pimento': 99, 'carrot': 0, 'cabbage1': 45, 'cabbage2': 2, 'eggplant': 40, 'bread': 36, 'baguette': 7, 'sandwich': 37, 'hamburger': 8, 'hotdog': 10, 'donuts': 84, 'cake': 58, 'onion': 82, 'marshmallow': 80, 'mooncake': 38, 'shirimpsushi': 60, 'sushi1': 62, 'sushi2': 61, 'big_spoon': 30, 'small_spoon': 112, 'fork': 27, 'knife': 93, 'big_plate': 115, 'small_plate': 11, 'bowl': 110, 'white_ricebowl': 109, 'blue_ricebowl': 118, 'black_ricebowl': 116, 'green_ricebowl': 119, 'black_mug': 34, 'gray_mug': 100, 'pink_mug': 108, 'green_mug': 117, 'blue_mug': 101, 'blue_cup': 33, 'orange_cup': 56, 'yellow_cup': 57, 'big_wineglass': 114, 'small_wineglass': 29, 'glass1': 111, 'glass2': 54, 'glass3': 107, 'bg': 120}

if __name__ == '__main__' :
    jm = json_maker([], json_path, 0)
    simple_label_list = get_list_from_file(simple_label_path)

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

        inst_id = jm.get_number_of_insts(scene_num, cam_num) + 1
        jm.insert_instance(scene_num, cam_num, inst_id, cls_idx, x1, y1, x2, y2, prob) 
    jm.sort()
    jm.save()


import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

from argparse import ArgumentParser

if __name__ == '__main__' :
    parser=ArgumentParser()
    parser.add_argument('--image_path', dest='image_path', default = '/data3/sap/mvcnn/svdet+triplenet', help = 'Path to images for classification')
    parser.add_argument('--save_path', dest= 'save_path', default= '/data3/sap/mvcnn/svdet+triplenet_labeled',help = 'Path(parent directory) to images for saving')
    parser.add_argument('--set', dest='set', default = 'test', help = 'specify dataset(train/val/test)')
    options = parser.parse_args()

    subcls_name_dict = {'water1': 0, 'water2': 1, 'pepsi': 2, 'coca1': 3, 'coca2': 4, 'coca3': 5, 'coca4': 6, 'tea1': 7, 'tea2': 8, 'yogurt': 9, 'ramen1': 10, 'ramen2': 11, 'ramen3': 12, 'ramen4': 13, 'ramen5': 14, 'ramen6': 15, 'ramen7': 16, 'juice1': 17, 'juice2': 18, 'can1': 19, 'can2': 20, 'can3': 21, 'can4': 22, 'can5': 23, 'can6': 24, 'can7': 25, 'can8': 26, 'can9': 27, 'ham1': 28, 'ham2': 29, 'pack1': 30, 'pack2': 31, 'pack3': 32, 'pack4': 33, 'pack5': 34, 'pack6': 35, 'snack1': 36, 'snack2': 37, 'snack3': 38, 'snack4': 39, 'snack5': 40, 'snack6': 41, 'snack7': 42, 'snack8': 43, 'snack9': 44, 'snack10': 45, 'snack11': 46, 'snack12': 47, 'snack13': 48, 'snack14': 49, 'snack15': 50, 'snack16': 51, 'snack17': 52, 'snack18': 53, 'snack19': 54, 'snack20': 55, 'snack21': 56, 'snack22': 57, 'snack23': 58, 'snack24': 59, 'green_apple': 60, 'red_apple': 61, 'tangerine': 62, 'lime': 63, 'lemon': 64, 'yellow_quince': 65, 'green_quince': 66, 'white_quince': 67, 'fruit1': 68, 'fruit2': 69, 'peach': 70, 'banana': 71, 'fruit3': 72, 'pineapple': 73, 'fruit4': 74, 'strawberry': 75, 'cherry': 76, 'red_pimento': 77, 'green_pimento': 78, 'carrot': 79, 'cabbage1': 80, 'cabbage2': 81, 'eggplant': 82, 'bread': 83, 'baguette': 84, 'sandwich': 85, 'hamburger': 86, 'hotdog': 87, 'donuts': 88, 'cake': 89, 'onion': 90, 'marshmallow': 91, 'mooncake': 92, 'shirimpsushi': 93, 'sushi1': 94, 'sushi2': 95, 'big_spoon': 96, 'small_spoon': 97, 'fork': 98, 'knife': 99, 'big_plate': 100, 'small_plate': 101, 'bowl': 102, 'white_ricebowl': 103, 'blue_ricebowl': 104, 'black_ricebowl': 105, 'green_ricebowl': 106, 'black_mug': 107, 'gray_mug': 108, 'pink_mug': 109, 'green_mug': 110, 'blue_mug': 111, 'blue_cup': 112, 'orange_cup': 113, 'yellow_cup': 114, 'big_wineglass': 115, 'small_wineglass': 116, 'glass1': 117, 'glass2': 118, 'glass3': 119, 'background': 120 }

    new_subcls_dict = dict()

    for key, val in subcls_name_dict.items():
        #subcls_name_dict[key] = val + 1
        #new_subcls_dict[val+1] = key

        new_subcls_dict[val] = key # for ReidResult
    print("after editing, dict items are following")

    for key, val in new_subcls_dict.items():
        print("subcls: {0}, index: {1}.".format(key,val))
    def createDirectory(_dir):
        try:
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        except OSError:
            print("Error: Failed to create the directory.")

    num_subcls = 121
    img_path = np.sort(np.asarray(glob.glob(os.path.join(options.image_path,'*.jpg'))))

    num_img = 0


    for imgpath in tqdm(img_path):
        img = cv2.imread(imgpath)
        img_resized = cv2.resize(img,(256,256))
        subcls = imgpath.split('_')[1].split('.')[0] #string type
        #subcls_name = subcls_name_dict[int(subcls)]
        subcls_name = new_subcls_dict[int(subcls)]
        view_num = imgpath.split('-')[3]
        
        instance = imgpath.split('-')[4].split('_')[0]
        
        #print(subcls+'\n')

        full_save_path = options.save_path + '/' + subcls_name + '/'+options.set+'/' + imgpath.rsplit('-',2)[0].split('/')[-1]+'-'+subcls_name+'-'+instance +'_' + str(view_num)
        createDirectory(options.save_path+ '/' +  subcls_name+'/'+options.set)
        cv2.imwrite(full_save_path+'.png',img_resized)
        #print('{0} is saved'.format(full_save_path+'.png'))
        num_img = num_img+1


    print("{} images are saved".format(num_img))


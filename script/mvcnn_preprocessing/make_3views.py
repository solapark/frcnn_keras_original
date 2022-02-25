import os 
import cv2
import numpy as np
import glob

from argparse import ArgumentParser
from tqdm import tqdm

if __name__ == '__main__' :
    parser=ArgumentParser()
    parser.add_argument('--image_path', dest='image_path', help = 'Path to images for classification')
#parser.add_argument('--save_path', dest='save_path', default = '/data3/sjyang/dataset/MVCNN_dataset', help = 'Path to images for saving')
    parser.add_argument('--set', dest = 'set_', choices = ['train', 'val', 'test'])


    options = parser.parse_args()

    dir_list = os.listdir(options.image_path)

    for class_name in tqdm(dir_list):
        #img_path = np.sort(np.asarray(glob.glob(os.path.join(options.image_path + class_name + '/'+ options.set_+'/','*.png'))))
        img_path = np.sort(np.asarray(glob.glob(os.path.join(options.image_path, class_name, options.set_,'*.png'))))
        for imgpath in tqdm(img_path):
            view_num = imgpath.split('_')[-1].split('.')[0]
            if(len(imgpath.split('_'))) == 3: 
                temp_path = imgpath.split('_')[0]+'_'+imgpath.split('_')[1]+'_'
                empty_flag = [os.path.exists(imgpath.split('_')[0]+'_'+imgpath.split('_')[1]+'_'+str(i)+'.png') for i in range (1,4)]
            else:
                temp_path = imgpath.split('_')[0]+'_'+imgpath.split('_')[1]+'_'+imgpath.split('_')[2]+'_'+imgpath.split('_')[3]+'_'
                empty_flag = [os.path.exists(temp_path+str(i)+'.png')for i in range(1,4)]
            num_empty = empty_flag.count(False)
            if num_empty == 1:
                #index_for_copy = empty_flag.index(True)
                temp_array = np.array(empty_flag)
                index_for_copy_list = np.where(temp_array == True)[0].tolist()
                np.random.shuffle(index_for_copy_list)
                index_for_copy = index_for_copy_list[0]
                index_empty = empty_flag.index(False)
                #img = cv2.imread(temp_path + str(index_for_copy+1)+'.png')
                #cv2.imwrite(temp_path+str(index_empty+1)+'.png',img)
                #print('img {0} is copied from img {1}'.format(temp_path+str(index_empty+1)+'.png',temp_path+str(index_for_copy+1)+'.png'))
                src = temp_path + str(index_for_copy+1)+'.png'
                dst = temp_path+str(index_empty+1)+'.png'
                os.symlink(src, dst) 
            elif num_empty == 2:
                index_for_copy = empty_flag.index(True)
                index_list_empty = [1,2,3]
                index_list_empty.remove(index_for_copy+1)
                #img = cv2.imread(temp_path+str(index_for_copy+1)+'.png')
                src = temp_path+str(index_for_copy+1)+'.png'
                for view_num in index_list_empty:
                    dst = temp_path+str(view_num)+'.png'
                    os.symlink(src, dst) 
                    #cv2.imwrite(temp_path+str(view_num)+'.png',img)
                    #print('img {0} is copied from img {1}'.format(temp_path + str(view_num)+'.png',temp_path+str(index_for_copy+1)+'.png'))
            #if((int(view_num) < 3) and not (os.path.exists(imgpath.split('_')[0]+'_'+imgpath.split('_')[1] + '_3.png'))):
            #    if(os.path.exists(imgpath.split('_')[0]+'_'+imgpath.split('_')[1]+'_2.png')):
            #        img = cv2.imread(imgpath.split('_')[0]+'_'+imgpath.split('_')[1]+'_2.png')
            #        cv2.imwrite(imgpath.split('_')[0]+'_'+imgpath.split('_')[1]+'3.png',img)
            #        print('copy img is {0}'.format(imgpath.split('_')[0]+'_'+imgpath.split('_')[1]+'_3.png'))
            #    elif(os.path.exists(imgpath.split('_')[0]+'_'+imgpath.split('_')[1]+'_1.png')):
            #        img = cv2.imread(imgpath.split('_')[0]+'_'+imgpath.split('_')[1]+'_1.png')
            #        cv2.imwrite(imgpath.split('_')[0]+'_'+imgpath.split('_')[1]+'_3.png',img)
            #        print('copy img is {0}'.format(imgpath.split('_')[0]+'_'+imgpath.split('_')[1]+'_3.png'))
#img_path = np.sort(np.asarray(glob.glob(options.image_path)))

#print(dir_list)

import cv2
import numpy as np

def get_concat_img(img_list, cols=3):
    rows = int(len(img_list)/cols)
    hor_imgs = [np.hstack(img_list[i*cols:(i+1)*cols]) for i in range(rows)]
    ver_imgs = np.vstack(hor_imgs)
    return ver_imgs

def draw_box(image, box, color = (0, 255, 0)):
    x1, y1, x2, y2 = box
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image 

def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w*h

def iou(a, b):
    # a and b should be (x1,y1,x2,y2)
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)

def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union

def get_file_list_from_dir(dir_path, exp='', is_full_path = True, file_form = ''):
    #use : get_file_list_from_dir("/home/sap", "png")
    file_form = dir_path + '/*' 
    if(exp):
        file_form += '.'+exp 
    file_list = glob.glob(file_form)
    if not is_full_path :
        new_file_list = [get_name_from_path(cur_file) for cur_file in file_list]
        file_list = new_file_list
    return file_list

def calc_emb_dist(embs1, embs2) : 
    '''
    calc emb dist for last axis
    Args :
        embs1 and embs2 have same shape (ex : (2, 3, 4))
    Return :
        dist for last axis (ex : (2, 3))
    '''
    return np.sqrt(np.sum(np.square(embs1 - embs2), -1)) 

def get_min_emb_dist_idx(emb, embs, thresh = np.zeros(0), is_want_dist = 0): 
    '''
    Args :
        emb (shape : m, n)
        embs (shape : m, k, n)
        thresh_dist : lower thersh. throw away too small dist (shape : m, )
    Return :
        min_dist_idx (shape : m, 1)
    '''
    emb_ref = emb[:, np.newaxis, :]
    dist = calc_emb_dist(emb_ref, embs) #(m, k)

    if(thresh.size) : 
        thresh = thresh[:, np.newaxis] #(m, 1)
        dist[dist<=thresh] = np.inf 
    min_dist_idx = np.argmin(dist, 1) #(m, )
    if(is_want_dist):
        min_dist = dist[np.arange(len(dist)), min_dist_idx]
        return min_dist_idx, min_dist
    return min_dist_idx

def get_new_img_size(width, height, img_min_side=300):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = img_min_side
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = img_min_side
    return resized_width, resized_height



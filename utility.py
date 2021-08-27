import os
import time
import datetime
import csv
import re
import math
import cv2
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

from gt import data_generators

def make_save_dir(base_dir, save_dir, reset):
    save_path = os.path.join(base_dir, 'experiment', save_dir)
    if(reset):
        model_path = os.path.join(save_path, 'model')

        if reset : os.system('rm -rf %s'%(save_path))

        os.makedirs(save_path, exist_ok = True)
        os.makedirs(model_path, exist_ok = True)
    return save_path

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

def mv_iou(boxes_a, boxes_b, is_valids_a, is_valids_b):
    area_i_list, area_u_list = [], []
    for box_a, box_b, is_valid_a, is_valid_b in zip(boxes_a, boxes_b, is_valids_a, is_valids_b):
        if is_valid_a and is_valid_b :
            area_i = intersection(box_a, box_b) 
        else :
            area_i = 0
            if not is_valid_a :
                box_a = np.zeros((4, ))
            if not is_valid_b :
                box_b = np.zeros((4, ))
        area_u = union(box_a, box_b, area_i)
        area_i_list.append(area_i)
        area_u_list.append(area_u)

    return float(sum(area_i_list)) / float(sum(area_u_list) + 1e-6), area_i_list, area_u_list

def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union

def get_concat_img(img_list, cols=3):
    rows = int(len(img_list)/cols)
    hor_imgs = [np.hstack(img_list[i*cols:(i+1)*cols]) for i in range(rows)]
    ver_imgs = np.vstack(hor_imgs)
    return ver_imgs

def write_config_sv(path, option, C, is_reset):
    if(is_reset):
        f = open(path, 'w')
    else :
        f = open(path, 'a')
    f.write(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '\n\n')
    for k, v in vars(C).items():
        f.write('{}: {}\n'.format(k, v))
    f.write('\n')
    for k, v in vars(option).items():
        f.write('{}: {}\n'.format(k, v))
    f.write('\n')
    f.close()

class Log_manager_sv:
    def __init__(self, save_dir, reset, header, file_name = 'log.csv'):
        self.path = os.path.join(save_dir, file_name)
        #if(reset): self.write(['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'time'])
        if(reset): self.write(header)
    
    def write(self, c):
        f = open(self.path, 'a')
        wr = csv.writer(f)
        wr.writerow(c)
        f.close()


class Model_path_manager:
    def __init__(self, args):
        self.args = args
        self.model_dir = os.path.join(args.base_path, 'experiment', args.save_dir, 'model')
        base_name = 'model.hdf5'
        self.name_pattern = "model_([0-9]*)\.hdf5"
        name_prefix, self.ext = base_name.split('.')

        self.prefix = os.path.join(self.model_dir, name_prefix)
        
        if args.resume :
            self.resume_epoch = self.get_resume_epoch()
            self.cur_epoch = self.resume_epoch + 1
        else :
            self.cur_epoch = 1
        
    def get_resume_epoch(self) :
        model_path_regex = get_value_in_pattern(self.args.input_weight_path, self.name_pattern)
        return int(model_path_regex)

    def get_save_path(self):
        save_path = '%s_%04d.%s' %(self.prefix, self.cur_epoch, self.ext)
        self.cur_epoch += 1 
        return save_path

    def get_all_path(self):
        all_names = os.listdir(self.model_dir)
        all_names.sort()
        return [os.path.join(self.model_dir, name) for name in all_names]

    def get_path(self, name) : 
        return '%s_%s.%s' %(self.prefix, name, self.ext)

    def get_path_in_range(self, start_idx, end_idx, interval = 1):
        all_paths = []

        for idx in range(start_idx, end_idx+1, interval):
            path = self.get_path(str(idx))
            all_paths.append(path)
        return all_paths
            

class Data_to_monitor :
    def __init__(self, name, names) :
        self.name = name
        self.names = names
        self.num_data = len(names)
        self.reset()

    def add(self, data):
        self.data = np.concatenate([self.data, np.array(data).reshape(-1, self.num_data)])

    def mean(self):
        return np.nanmean(self.data, axis=0)

    def reset(self):
        self.data = np.zeros((0, self.num_data))

    def get_name(self):
        return self.names

    def get_best(self):
        best_idx = np.argmax(self.data)
        best = self.data[best_idx]
        return best_idx[0], best

    def get_length(self):
        return len(self.data)

    def get_data(self):
        return self.data

    def load(self, path):
        self.data = np.load(path).reshape((-1, self.num_data))

    def save(self, path):
        np.save(path, self.data)

    def plot(self, path):
        epoch = len(self.data)
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title(self.name)
        for n, d in zip(self.names, self.data.T) :
            plt.plot(axis, d, label=n)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(self.name)
        plt.grid(True)
        plt.savefig(path)
        plt.close(fig)

    def display(self):
        log = ['%s: %.4f'%(n, v) for n, v in zip(self.names, self.mean())]
        return ' '.join(log)
        
class Log_manager:
    def __init__(self, args):
        self.args = args
        self.dir = os.path.join(args.base_path, 'experiment', args.save_dir)

        self.write_config()
        self.log_file = self.get_log_file()

        if args.mode == 'train' :
            self.loss_every_iter = Data_to_monitor('Loss', args.loss_log)
            self.loss_every_epoch = Data_to_monitor('Loss', args.loss_log)
            self.num_calssifier_pos_samples_every_iter = Data_to_monitor('num_calssifier_pos_samples', ['num_calssifier_pos_samples'])
            self.num_calssifier_pos_samples_every_epoch = Data_to_monitor('num_calssifier_pos_samples',['num_calssifier_pos_samples'])

            if args.resume:
                self.loss_every_epoch.load(self.get_path('loss.npy'))
                self.num_calssifier_pos_samples_every_epoch.load(self.get_path('num_calssifier_pos_samples.npy'))
                print('Continue from epoch {}...'.format(len(self.loss_every_epoch.get_data())))

        if args.mode in ['val', 'val_models'] :
            self.ap_names = args.class_list
            self.ap = Data_to_monitor('ap', self.ap_names[:-1])
            self.map = Data_to_monitor('map', ['map'])
            self.iou = Data_to_monitor('iou', ['iou'])

            self.best_map = 0
            self.best_map_epoch = 0

            if args.resume:
                self.ap.load(self.get_path('ap.npy'))
                self.map.load(self.get_path('map.npy'))
                self.iou.load(self.get_path('iou.npy'))
                
                self.best_map_epoch, self.best_map = slef.map.get_best()

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def get_log_file(self):
        self.log_file_name = 'log_%s.txt' % (self.args.mode)
        open_type = 'a' if os.path.exists(self.get_path(self.log_file_name))else 'w'
        log_file = open(self.get_path(self.log_file_name), open_type)
        return log_file

    def write_config(self) :
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.config_file_name = 'config_%s.txt' % (self.args.mode)
        open_type = 'a' if os.path.exists(self.get_path(self.config_file_name))else 'w'
        with open(self.get_path(self.config_file_name), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(self.args):
                f.write('{}: {}\n'.format(arg, getattr(self.args, arg)))
            f.write('\n')

    def save(self):
        if(self.args.mode == 'train'):
            self.loss_every_epoch.save(self.get_path('loss'))
            self.loss_every_epoch.plot(self.get_path('loss.pdf'))
            self.num_calssifier_pos_samples_every_epoch.save(self.get_path('num_calssifier_pos_samples'))
            self.num_calssifier_pos_samples_every_epoch.plot(self.get_path('num_calssifier_pos_samples.pdf'))
        if(self.args.mode in ['val', 'val_models']):
            self.ap.save(self.get_path('ap'))
            self.map.save(self.get_path('map'))
            self.iou.save(self.get_path('iou'))
            
            self.ap.plot(self.get_path('ap.pdf'))
            self.map.plot(self.get_path('map.pdf'))
            self.iou.plot(self.get_path('iou.pdf'))

    def add(self, data, name):
        if name == 'num_calssifier_pos_samples':
            self.num_calssifier_pos_samples_every_iter.add(data)
        if name == 'loss':
            self.loss_every_iter.add(data)
        elif name == 'ap':
            self.ap.add(data)
            mAP = sum(data)/len(data)
            if(mAP > self.best_map):
                self.best_map = mAP
                self.best_map_epoch = self.map.get_length() + 1
            self.map.add(mAP)
        
        elif name == 'iou' :
            self.iou.add(data)

    def epoch_done(self):
        self.loss_every_epoch.add(self.loss_every_iter.mean())
        self.num_calssifier_pos_samples_every_epoch.add(self.num_calssifier_pos_samples_every_iter.mean())

        self.loss_every_iter.reset()
        self.num_calssifier_pos_samples_every_iter.reset()
        self.save()

    def display(self, name):
        if name == 'loss':
            result = self.loss_every_iter.display()
        elif name == 'num_calssifier_pos_samples':
            result = self.num_calssifier_pos_samples_every_iter.display()
        return result

    def write_log(self, log, refresh=True):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path(self.log_file_name), 'a')

    def write_cur_time(self):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.write_log(now)

    def done(self):
        self.log_file.close()

    def plot(self, data, label):
        epoch = len(data)
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title(label)
        if(label == 'loss'):
            for loss_name, loss in zip(self.loss_names, self.loss.T) :
                plt.plot( axis, loss, label=loss_name)
        elif(label == 'ap'):
            for ap_name, ap in zip(self.ap_names, self.ap.T) :
                plt.plot( axis, ap, label=ap_name)
        else :
            plt.plot( axis, data, label=label)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(label)
        plt.grid(True)
        plt.savefig(self.get_path('%s.pdf')%(label))
        plt.close(fig)

    def get_best_map(self):
        return self.best_map

    def get_best_map_idx(self):
        return self.best_map_idx

class Result_img_manager :
    def __init__ (self, args):
        os.makedirs(self.get_path('results-{}'.format(test_name)), exist_ok=True)
        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

def get_map(pred, gt, f):
	T = {}
	P = {}
	iou_result = 0
	fx, fy = f

	for bbox in gt:
		bbox['bbox_matched'] = False

	pred_probs = np.array([s['prob'] for s in pred])
	#print(pred)
	#print(pred_probs)
	box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

	for box_idx in box_idx_sorted_by_prob:
		pred_box = pred[box_idx]
		pred_class = pred_box['class']
		pred_x1 = pred_box['x1']
		pred_x2 = pred_box['x2']
		pred_y1 = pred_box['y1']
		pred_y2 = pred_box['y2']
		pred_prob = pred_box['prob']
		if pred_class not in P:
			P[pred_class] = []
			T[pred_class] = []
		P[pred_class].append(pred_prob)
		found_match = False

		for gt_box in gt:
			gt_class = gt_box['class']
			gt_x1 = gt_box['x1']/fx
			gt_x2 = gt_box['x2']/fx
			gt_y1 = gt_box['y1']/fy
			gt_y2 = gt_box['y2']/fy
			gt_seen = gt_box['bbox_matched']
			if gt_class != pred_class:
				continue
			if gt_seen:
				continue
			iou = 0
			iou = data_generators.iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
			iou_result += iou
			#print('IoU = ' + str(iou))
			if iou >= 0.5:
				found_match = True
				gt_box['bbox_matched'] = True
				break
			else:
				continue

		T[pred_class].append(int(found_match))
	for gt_box in gt:
		if not gt_box['bbox_matched']: # and not gt_box['difficult']:
			if gt_box['class'] not in P:
				P[gt_box['class']] = []
				T[gt_box['class']] = []

			T[gt_box['class']].append(1)
			P[gt_box['class']].append(0)

	#import pdb
	#pdb.set_trace()
	return T, P, iou_result

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

def get_new_img_size(width, height, img_min_side):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = img_min_side
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = img_min_side
    return resized_width, resized_height

class Map_calculator:
    #def __init__(self, args, fx, fy):
    def __init__(self, args):
        self.num_valid_cam = args.num_valid_cam
        #self.fx, self.fy = fx, fy         
        self.class_list_wo_bg = args.class_list[:-1]
        self.reset()

    def reset(self):
        self.all_T = {cls : [] for cls in self.class_list_wo_bg}
        self.all_P = {cls : [] for cls in self.class_list_wo_bg}
        self.iou_result = 0
        self.cnt = 0

    '''
    def get_gt_batch(self, labels_batch):
        return [self.get_gt(labels) for labels in labels_batch]

    def get_gt(self, labels):
        gts = [[] for _ in range(self.num_valid_cam)]
        for inst in labels :
            gt_cls = self.class_list_wo_bg[inst['cls']]
            gt_boxes = inst['resized_box']
            for cam_idx in range(self.num_valid_cam) : 
                if cam_idx in gt_boxes :
                    x1, y1, x2, y2 = gt_boxes[cam_idx]
                    info = {'class':gt_cls, 'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}
                    gts[cam_idx].append(info)
        return gts
    '''

    def get_iou(self):
        return self.iou_result/self.cnt

    def get_map(self) :
        return np.mean(np.array(self.all_aps))

    def get_aps(self):
        all_aps = [average_precision_score(t, p) if t else 0 for t, p in zip(self.all_T.values(), self.all_P.values())]
        all_aps = [ap if not math.isnan(ap) else 0 for ap in all_aps]
        self.all_aps = all_aps
        return all_aps

    def get_aps_dict(self):
        return {cls : ap for cls, ap in zip(self.class_list_wo_bg, self.all_aps)}

    def add_img_tp(self, dets, gts):
        self.cnt += 1
        T, P, iou = self.get_img_tp(dets, gts)
        self.iou_result += iou
        for key in T.keys():
            self.all_T[key].extend(T[key])
            self.all_P[key].extend(P[key])

    def get_img_tp(self, pred, gt):
        T = {}
        P = {}
        iou_result = 0

        for bbox in gt:
            bbox['bbox_matched'] = False

        pred_probs = np.array([s['prob'] for s in pred])
        #print(pred)
        #print(pred_probs)
        box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

        for box_idx in box_idx_sorted_by_prob:
            pred_box = pred[box_idx]
            pred_class = pred_box['class']
            pred_x1 = pred_box['x1']
            pred_x2 = pred_box['x2']
            pred_y1 = pred_box['y1']
            pred_y2 = pred_box['y2']
            pred_prob = pred_box['prob']
            if pred_class not in P:
                P[pred_class] = []
                T[pred_class] = []
            P[pred_class].append(pred_prob)
            found_match = False

            for gt_box in gt:
                gt_class = gt_box['class']
                gt_x1 = gt_box['x1']
                gt_x2 = gt_box['x2']
                gt_y1 = gt_box['y1']
                gt_y2 = gt_box['y2']
                gt_seen = gt_box['bbox_matched']
                if gt_class != pred_class:
                    continue
                if gt_seen:
                    continue
                iou = 0
                iou = data_generators.iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
                iou_result += iou
                #print('IoU = ' + str(iou))
                if iou >= 0.5:
                    found_match = True
                    gt_box['bbox_matched'] = True
                    break
                else:
                    continue

            T[pred_class].append(int(found_match))
        for gt_box in gt:
            if not gt_box['bbox_matched']: # and not gt_box['difficult']:
                if gt_box['class'] not in P:
                    P[gt_box['class']] = []
                    T[gt_box['class']] = []

                T[gt_box['class']].append(1)
                P[gt_box['class']].append(0)

        #import pdb
        #pdb.set_trace()
        return T, P, iou_result


class Img_preprocessor:
    def __init__ (self, args):
        self.num_valid_cam = args.num_valid_cam
        self.resized_width, self.resized_height = args.resized_width, args.resized_height

        '''
        if args.width <= args.height:
            self.f = args.im_size/args.width
        else:
            self.f = args.im_size/args.height
        self.fx = args.width/float(self.resized_width)
        self.fy = args.height/float(self.resized_height)
        '''

        self.img_channel_mean0 = args.img_channel_mean[0]
        self.img_channel_mean1 = args.img_channel_mean[1]
        self.img_channel_mean2 = args.img_channel_mean[2]
        self.img_scaling_factor = args.img_scaling_factor
    def process_batch(self, batch):
        #batch (num_valid_cam, batch, w, h, c)
        result_batch = []
        for b in np.array(batch).transpose(1, 0, 2, 3, 4) :
            processed_b = [self.process_img(img) for img in b]
            result_batch.append(processed_b)
        return np.array(result_batch).transpose(1, 0, 2, 3, 4)
        
    def process_img(self, img):
        img = img[:, :, (2, 1, 0)]
        img = img.astype(np.float32)
        img[:, :, 0] -= self.img_channel_mean0
        img[:, :, 1] -= self.img_channel_mean1
        img[:, :, 2] -= self.img_channel_mean2
        #img /= self.img_scaling_factor
        #img = np.transpose(img, (2, 0, 1))
        return img


class Sv_gt_batch_generator:
    def __init__(self, args):
        self.num_valid_cam = args.num_valid_cam
        self.class_list_wo_bg = args.class_list[:-1]

    def get_gt_batch(self, mv_labels_batch):
        return [self.get_gt(mv_labels) for mv_labels in mv_labels_batch]

    def get_gt(self, mv_labels):
        gts = [[] for _ in range(self.num_valid_cam)]
        for inst in mv_labels :
            gt_cls = self.class_list_wo_bg[inst['cls']]
            gt_boxes = inst['resized_box']
            for cam_idx in range(self.num_valid_cam) : 
                if cam_idx in gt_boxes :
                    x1, y1, x2, y2 = gt_boxes[cam_idx]
                    info = {'class':gt_cls, 'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}
                    gts[cam_idx].append(info)
        return gts

class CALC_REGR:
    def __init__(self, std):
        self.std = np.array(std).reshape(-1, 1)

    def calc_t(self, pred, gt):
        (cx_gt, cy_gt), (cx_pred, cy_pred) = map(lambda a : [(a[:, 0]+a[:, 2])/2.0, (a[:, 1]+a[:, 3])/2.0], [gt, pred])
        (w_gt, h_gt), (w_pred, h_pred) = map(lambda a : [a[:, 2]-a[:, 0], a[:, 3]-a[:, 1]], [gt, pred])

        tx = (cx_gt - cx_pred) / w_pred
        ty = (cy_gt - cy_pred) / h_pred
        tw = np.log(w_gt/w_pred)
        th = np.log(h_gt/h_pred)

        tx, ty, tw, th = self.std * [tx, ty, tw, th]

        return np.column_stack([tx, ty, tw, th])

def pickle_save(l, path):
    f = open(path, 'wb')
    pickle.dump(l,f)

def pickle_load(path) :
    f = open(path, 'rb')
    return pickle.load(f)
   
def file_system(args):
    if args.reset and args.mode == 'train' :
        save_path = os.path.join(args.base_dir, 'experiment', args.save_dir)
        model_path = os.path.join(save_path, 'model')

        os.system('rm -rf %s'%(save_path))
        os.makedirs(save_path, exist_ok = True)
        os.makedirs(model_path, exist_ok = True)

def draw_cls_box_prob(img_list, bboxes, probs, ious, args, num_cam = 1,is_nms=True) : 
    height,_,_ = img_list[0].shape
    img_min_side = float(args.im_size)
    ratio = img_min_side/height

    for key in sorted(bboxes):
        bbox = np.array(bboxes[key]) #(num_cam, num_box, 4)
        if(is_nms):
            new_boxes_all_cam, new_probs = non_max_suppression_fast_multi_cam(bbox, np.array(probs[key]), overlap_thresh=0.5)
        else : 
            new_boxes_all_cam, new_probs, new_ious_all_cam = bbox, np.array(probs[key]), np.array(ious[key])
        instance_to_color = [np.random.randint(0, 255, 3) for _ in range(len(new_probs))]
        for jk in range(new_boxes_all_cam.shape[1]):
            new_boxes = new_boxes_all_cam[:, jk] #(num_cam, 4)
            new_ious = new_ious_all_cam[jk, :] #(num_cam, )
            all_dets = []
            result_img_list = []
            for cam_idx in range(num_cam) : 
                img = img_list[cam_idx].copy()
                (x1, y1, x2, y2) = new_boxes[cam_idx]
                if(x1 == -args.rpn_stride) :
                    result_img_list.append(np.copy(img))
                    continue 
                # Calculate real coordinates on original image
                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                color = (int(instance_to_color[jk][0]), int(instance_to_color[jk][1]), int(instance_to_color[jk][2])) 
                cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), color, 4)
                iou = new_ious[cam_idx]
                textLabel = '{}: {}, {}'.format(key,int(100*new_probs[jk]), int(100*iou))
                all_dets.append((key,100*new_probs[jk], iou))
                (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                textOrg = (real_x1, real_y1)
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 1)
                cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

            img_list[cam_idx] = img
    print(all_dets)
    conc_img = get_concat_img(img_list, cols=3)
    cv2.imshow('cls_result', conc_img)
    cv2.waitKey()
    '''
    plt.figure(figsize=(10,10))
    for i, img in enumerate(img_list) : 
        coord = int('1'+str(num_cam)+str(i+1))
        plt.subplot(coord)
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.show()
    '''

def classifier_output_to_box_prob(ROIs_list, P_cls, P_regr, is_valids_all_cam, args, bbox_threshold, num_cam, is_demo, is_exclude_bg=False) : 
    if ROIs_list.ndim == 3 :
        ROIs_list = np.expand_dims(ROIs_list, 0)
    #class_mapping = args.num2cls
    class_mapping = args.class_list
    bboxes = {}
    probs = {}
    is_valids = {}
    # Calculate bboxes coordinates on resized image
    for ii in range(P_cls.shape[1]):
        # Ignore 'bg' class
        if is_exclude_bg and np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1) : 
            continue

        if np.max(P_cls[0, ii, :]) < bbox_threshold and is_demo :
            continue

        cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

        if cls_name not in bboxes:
            bboxes[cls_name] = [[] for _ in range(num_cam)]
            is_valids[cls_name] = [[] for _ in range(num_cam)]
            probs[cls_name] = []

        cls_num = np.argmax(P_cls[0, ii, :])
        
        cam_offset = (len(args.class_mapping) - 1) * 4
        for cam_idx in range(num_cam) : 
            #(x, y, w, h) = ROIs[0, ii, :]
            (x, y, w, h) = ROIs_list[cam_idx][0, ii, :]
            try:
                #(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                (tx, ty, tw, th) = P_regr[0, ii, cam_offset*cam_idx + 4*cls_num : cam_offset*cam_idx + 4*(cls_num+1)]
                tx /= args.classifier_std_scaling[0]
                ty /= args.classifier_std_scaling[1]
                tw /= args.classifier_std_scaling[2]
                th /= args.classifier_std_scaling[3]
                x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name][cam_idx].append([args.rpn_stride*x, args.rpn_stride*y, args.rpn_stride*(x+w), args.rpn_stride*(y+h)])
            is_valids[cls_name][cam_idx].append(is_valids_all_cam[0, ii, cam_idx])
        probs[cls_name].append(np.max(P_cls[0, ii, :]))
    return bboxes, probs, is_valids 

def get_real_coordinates(ratio, x1, y1, x2, y2):

    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2 ,real_y2)

def draw_nms(nms_list, debug_img, rpn_stride) : 
    nms_np = np.stack(nms_list, 0) #(num_cam, 300, 4)
    nms_np = nms_np.astype(int)*rpn_stride
    for cam_idx, nms in enumerate(nms_np) :
        img = np.copy(debug_img[cam_idx]).squeeze()
        window_name = 'nms' + str(cam_idx)
        for box in nms:
            img = draw_box(img, box, window_name, is_show=False)
        img = cv2.resize(img, (320, 180))
        cv2.imshow(window_name, img)
    cv2.waitKey()
 
def draw_box(image, box, name, color = (0, 255, 0), is_show = True, text = None):
    image = np.copy(image).squeeze()
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    if text:
        cv2.putText(image, text, (x1+10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        
    image = cv2.resize(image, (320, 180))
    if is_show :
        cv2.imshow(name, image)
    return image

def draw_line(img, pt1, pt2, color = (0, 255, 0)):
    img = np.copy(img)
    img = cv2.line(img, pt1, pt2, color, 5)
    return img

def calc_emb_dist(embs1, embs2) : 
    '''
    calc emb dist for last axis
    Args :
        embs1 and embs2 have same dims  (ex : (2, 1, 4) and (2, 3, 4))
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
    #emb_ref = emb[:, np.newaxis, :]
    emb_ref = np.expand_dims(emb, -2)
    dist = calc_emb_dist(emb_ref, embs) #(m, k)

    if(thresh.size) : 
        thresh = thresh[:, np.newaxis] #(m, 1)
        dist[dist<=thresh] = np.inf 
    min_dist_idx = np.argmin(dist, -1) #(m, )
    if(is_want_dist):
        min_dist = dist[np.arange(len(dist)), min_dist_idx] if dist.ndim > 1 else dist[min_dist_idx]
        return min_dist_idx, min_dist
    return min_dist_idx

def non_max_suppression_fast_multi_cam(boxes, probs, is_valids, overlap_thresh=0.9, max_boxes=300):
    # boxes : (num_cam, num_box, 4)
    # probs : (num_box, )
    # is_valids : (num_cam, num_box)
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list

    # Process explanation:
    #   Step 1: Sort the probs list
    #   Step 2: Find the larget prob 'Last' in the list and save it to the pick list
    #   Step 3: Calculate the IoU with 'Last' box and other boxes in the list. If the IoU is larger than overlap_threshold, delete the box from list
    #   Step 4: Repeat step 2 and step 3 until there is no item in the probs list 
    if len(boxes) == 0:
        return []

    boxes = boxes.transpose(1, 0, 2) #(num_box, num_cam, 4)
    is_valids = is_valids.transpose(1, 0) #(num_box, num_cam)
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, :, 0] #(num_box, num_cam)
    y1 = boxes[:, :, 1]
    x2 = boxes[:, :, 2]
    y2 = boxes[:, :, 3]

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes 
    pick = []

    # calculate the areas 
    area = (x2 - x1) * (y2 - y1) #(num_box, num_cam)

    # sort the bounding boxes 
    idxs = np.argsort(probs) #(num_box,)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the intersection
        xx1_int = np.maximum(x1[i], x1[idxs[:last]]) #x1[i]: (num_cam, ), x1[idxs[:last]]: (num_box, num_cam) #out: (num_box, num_cam)
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int) #(num_box, num_cam)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int #(num_box, num_cam)

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int/(area_union + 1e-6) #(num_box, num_cam)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            #np.where(np.all(overlap > overlap_thresh, 1))[0])))
            #np.where(np.any(overlap > overlap_thresh, 1))[0])))
            np.where(np.sum(overlap > overlap_thresh, 1) > 1)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int").transpose(1, 0, 2) #(num_cam, num_box, 4)
    is_valids = is_valids[pick].transpose(1, 0)
    probs = probs[pick]
    return boxes, probs, is_valids

def draw_reid(boxes_batch, is_valid_batch, debug_img, rpn_stride) : 
    boxes_list = boxes_batch[0].astype(int)*rpn_stride  #(300, num_cam, 4)
    is_valids_list = is_valid_batch[0] #(300, num_cam)

    boxes_list = boxes_list[::-1]
    is_valids_list = is_valids_list[::-1]
    for boxes, is_valids in zip(boxes_list, is_valids_list) :
        img_list = []
        for cam_idx, (box, is_valid)  in enumerate(zip(boxes, is_valids)):
            color = (0, 0, 255) if is_valid  else (0, 0, 0) 
            window_name = 'reid' + str(cam_idx)
            img = draw_box(np.copy(debug_img[cam_idx]), box, window_name, color, is_show=False, text=str(int(is_valid)))
            img_list.append(cv2.resize(img, (480, 240)))
        conc_img = get_concat_img(img_list)
        cv2.imshow('reid_result', conc_img)
        cv2.waitKey(0)

def draw_inst(img, x1, y1, x2, y2, cls, color, prob=None, inst_num = None):
    cv2.rectangle(img,(x1, y1), (x2, y2), color, 4)
    textLabel = '{}:{}'.format(cls,int(100*prob)) if prob else cls
    textLabel = '{}_{}'.format(inst_num,textLabel) if inst_num else textLabel
    (text_w,text_h) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_DUPLEX,1,1)
    textOrg = (x1-10, y1-text_h)
    cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

    return img
 
class Result_saver :
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.resize_ratio = args.im_size/args.height

        self.result_img_save_dir = os.path.join(args.base_path, 'experiment', args.save_dir, 'test_result')
        os.path.join(self.args.base_path, self.result_img_save_dir)
        os.makedirs(self.result_img_save_dir, exist_ok=True) 
        self.colors = np.random.randint(0, 255, (args.num_nms, 3)).tolist()

    def save(self, X, Y, img_paths, all_dets):
        save_path = self.get_general_file_name(img_paths)
        img_list = [x[0] for x in X]
        for cam_idx in range(self.args.num_valid_cam): 
            dets = all_dets[cam_idx]
            for det in dets:
                x1, y1, x2, y2, prob, cls, inst_idx = det['x1'], det['y1'], det['x2'], det['y2'], det['prob'], det['class'], det['inst_idx']
                det_color = self.colors[inst_idx]
                img_list[cam_idx] = draw_inst(img_list[cam_idx], x1, y1, x2, y2, cls, det_color, prob, inst_idx)
        conc_img = get_concat_img(img_list)
        cv2.imwrite(save_path, conc_img)
 
    def get_general_file_name(self, img_paths):
        base_name = os.path.basename(img_paths[0])
        if(self.args.dataset == 'MESSYTABLE'):
            general_name = get_value_in_pattern(base_name, '(.*)-0[0-9].jpg')
        general_path = os.path.join(self.result_img_save_dir, general_name+'.jpg')
        return general_path

def get_value_in_pattern(text, pattern):
    #pattern = 'aaa_(.*)'
    #text = 'aaa_b'
    #return = ['b']
    return re.findall(pattern, text)[0]


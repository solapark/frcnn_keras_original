import os
import time
import datetime
import csv
import re
import math
import cv2
import numpy as np
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

'''
def write_config(path, option, C, is_reset):
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

class Log_manager:
    def __init__(self, save_dir, reset, header, file_name = 'log.csv'):
        self.path = os.path.join(save_dir, file_name)
        #if(reset): self.write(['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'time'])
        if(reset): self.write(header)
    
    def write(self, c):
        f = open(self.path, 'a')
        wr = csv.writer(f)
        wr.writerow(c)
        f.close()

'''

class Model_path_manager:
    def __init__(self, args):
        self.model_dir = os.path.join(args.base_path, 'experiment', args.save_dir, 'model')
        base_name = 'model.hdf5'
        self.name_pattern = "model_([0-9]*)\.hdf5"
        name_prefix, self.ext = base_name.split('.')

        if(self.ext != 'hdf5'):
            print('Output weights must have .hdf5 filetype')
            exit(1)
        self.prefix = os.path.join(self.model_dir, name_prefix)
        
        if args.resume :
            self.resume_epoch = self.get_resume_epoch()
            self.cur_epoch = self.resume_epoch + 1
        else :
            self.cur_epoch = 1
        
    def get_resume_epoch(self) :
        filenames = os.listdir(self.model_dir)
        epoch_list = []
        for filename in filenames :
            model_path_regex = re.match(self.name_pattern, filename)
            epoch_list.append(int(model_path_regex.group(1)))
        return max(epoch_list)

    def get_resume_path(self):
        return '%s_%04d.%s' %(self.prefix, self.resume_epoch, self.ext)

    def get_save_path(self):
        save_path = '%s_%04d.%s' %(self.prefix, self.cur_epoch, self.ext)
        self.cur_epoch += 1 
        return save_path

    def get_all_path(self):
        all_names = os.listdir(self.model_dir)
        all_names.sort()
        return [os.path.join(self.model_dir, name) for name in all_names]

class Log_manager:
    def __init__(self, args):
        self.args = args
        self.dir = os.path.join(args.base_path, 'experiment', args.save_dir)

        self.write_config()
        self.log_file = self.get_log_file()

        if args.mode == 'train' :
            self.loss_names = args.loss_log
            if args.resume:
                self.loss = np.load(self.get_path('loss.npy'))
                print('Continue from epoch {}...'.format(len(self.loss)))

            elif args.reset:
                self.loss = np.zeros((0, len(self.loss_names)))

        elif args.mode in ['val', 'val_models'] :
            self.ap_names = args.class_list
            self.ap_names[-1] = 'mAP'
            self.ap = np.zeros((0,len(self.ap_names)))
            self.iou = np.zeros((0,))

            self.best_map = 0
            self.best_map_epoch = 0 

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def get_log_file(self):
        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        log_file = open(self.get_path('log.txt'), open_type)
        return log_file


    def write_config(self) :
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        open_type = 'a' if os.path.exists(self.get_path('config.txt'))else 'w'
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(self.args):
                f.write('{}: {}\n'.format(arg, getattr(self.args, arg)))
            f.write('\n')

    def save(self):
        if(self.args.mode == 'train'):
            np.save(self.get_path('loss'), self.loss)
            self.plot(self.loss, 'loss')
        elif(self.args.mode == 'val'):
            np.save(self.get_path('iou'), self.iou)
            np.save(self.get_path('ap'), self.ap)
            self.plot(self.iou, 'iou')
            self.plot(self.ap, 'ap')

    def add(self, data, name):
        if name == 'loss':
            self.loss = np.concatenate([self.loss, np.expand_dims(data, 0)])
        elif name == 'ap':
            mAP = sum(data)/len(data)
            data.append(mAP) #map
            self.ap = np.concatenate([self.ap, np.expand_dims(data, 0)])
            if(mAP > self.best_map):
                self.best_map = mAP
                self.best_map_epoch = len(self.ap) 
        
        elif name == 'iou' :
            self.iou = np.append(self.iou, data)

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

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
        self.num_cam = args.num_cam
        #self.fx, self.fy = fx, fy         
        self.class_list_wo_bg = args.class_list[:-1]
        self.reset()

    def reset(self):
        self.all_T = {cls : [] for cls in self.class_list_wo_bg}
        self.all_P = {cls : [] for cls in self.class_list_wo_bg}
        self.iou_result = 0
        self.cnt = 0

    def get_gt_batch(self, labels_batch):
        return [self.get_gt(labels) for labels in labels_batch]

    def get_gt(self, labels):
        gts = [[] for _ in range(self.num_cam)]
        for inst in labels :
            gt_cls = self.class_list_wo_bg[inst['cls']]
            gt_boxes = inst['resized_box']
            for cam_idx in range(self.num_cam) : 
                if cam_idx in gt_boxes :
                    x1, y1, x2, y2 = gt_boxes[cam_idx]
                    info = {'class':gt_cls, 'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}
                    gts[cam_idx].append(info)
        return gts


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
        self.num_cam = args.num_cam
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
        #batch (num_cam, batch, w, h, c)
        result_batch = []
        for b in np.array(batch).transpose(1, 0, 2, 3, 4) :
            processed_b = [self.process_img(img) for img in b]
            result_batch.append(processed_b)
        return np.array(result_batch).transpose(1, 0, 2, 3, 4)
        
    def process_img(self, img):
        img = cv2.resize(img, (self.resized_width, self.resized_height), interpolation=cv2.INTER_CUBIC)
        img = img[:, :, (2, 1, 0)]
        img = img.astype(np.float32)
        img[:, :, 0] -= self.img_channel_mean0
        img[:, :, 1] -= self.img_channel_mean1
        img[:, :, 2] -= self.img_channel_mean2
        img /= self.img_scaling_factor
        #img = np.transpose(img, (2, 0, 1))
        return img



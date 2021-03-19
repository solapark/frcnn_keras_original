from option import args
from parser import Parser
from model import Model
import utility

from keras.utils import generic_utils

from sklearn.metrics import average_precision_score
import os
import cv2
import numpy as np
import math


def format_img(img, C):
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
	
	if width <= height:
		f = img_min_side/width
		new_height = int(f * height)
		new_width = int(img_min_side)
	else:
		f = img_min_side/height
		new_width = int(f * width)
		new_height = int(img_min_side)
	fx = width/float(new_width)
	fy = height/float(new_height)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img, fx, fy

'''
def train(args, parser, model):
    save_path = utility.make_save_dir(args)
    model_path_manager = utility.Model_path_manager(args) 
	train_imgs, classes_count, _ = get_data(args.train_path)
    print('Training images per class:')
    pprint.pprint(classes_count)
    print(f'Num classes (including bg) = {len(classes_count)}')

    random.shuffle(train_imgs)
    num_imgs = len(train_imgs)
    print(f'Num train samples {len(train_imgs)}')

    data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, args, nn.get_img_output_length, K.common.image_dim_ordering(), mode='train')

    epoch_length = 1000
    iter_num = 0

    train_log = utility.Log_manager(args)

    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()

    best_loss = np.Inf
    for epoch_num in range(args.num_epochs):
        progbar = generic_utils.Progbar(epoch_length)
        print(f'Epoch {epoch_num + 1}/{args.num_epochs}')
        while True:
            try:
                if len(rpn_accuracy_rpn_monitor) == epoch_length :
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    print(f'Average number of overlapping bounding boxes from RPN = {mean_overlapping_bboxes} for {epoch_length} previous iterations')
                    if mean_overlapping_bboxes == 0:
                        print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

                X, Y, img_data = next(data_gen_train)
                loss_rpn = model_rpn.train_on_batch(X, Y)

                P_rpn = model_rpn.predict_on_batch(X)
                R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], args, K.common.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, args, class_mapping)

                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue

                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []
                
                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if args.num_rois > 1:
                    if len(pos_samples) < args.num_rois//2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, args.num_rois//2, replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, args.num_rois - len(selected_pos_samples), replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, args.num_rois - len(selected_pos_samples), replace=True).tolist()

                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                progbar.update(iter_num+1, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),
                                          ('detector_cls', losses[iter_num, 2]), ('detector_regr', losses[iter_num, 3])])

                iter_num += 1
                
                if iter_num == epoch_length:
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    class_acc = np.mean(losses[:, 4])

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    print(f'Mean number of bounding boxes from RPN overlapping ground truth boxes: {mean_overlapping_bboxes}')
                    print(f'Classifier accuracy for bounding boxes from RPN: {class_acc}')
                    print(f'Loss RPN classifier: {loss_rpn_cls}')
                    print(f'Loss RPN regression: {loss_rpn_regr}')
                    print(f'Loss Detector classifier: {loss_class_cls}')
                    print(f'Loss Detector regression: {loss_class_regr}')
                    print(f'Elapsed time: {time.time() - start_time}')

                    train_log.write([mean_overlapping_bboxes, class_acc, loss_rpn_cls, loss_rpn_regr, loss_class_cls, loss_class_regr, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')])

                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0
                    start_time = time.time()

                    model_all.save_weights(model_path_manager.get_save_path())
                    break

            except Exception as e:
                print(f'Exception: {e}')
                continue
    print('Training complete, exiting.')
'''

def val(args, parser, model):
    class_list_wo_bg = args.class_list[:-1]
    log_manager = utility.Log_manager(args)
    test_imgs, _, _ = parser.get_data(args.val_path)
    new_test_imgs = []
    '''
    for i in range(0, len(test_imgs), 40*3):
        new_test_imgs.extend(test_imgs[i:i+3])
    test_imgs = new_test_imgs
    #test_imgs = test_imgs
    '''
    
    model_path_manager = utility.Model_path_manager(args)
    #all_model_path = model_path_manager.get_all_path()
    all_model_path = model_path_manager.get_all_path()[-2:-1]
    num_models = len(all_model_path)
    for i, model_path in enumerate(all_model_path) :
        print('%d/%d calc map of %s'%(i+1, num_models, model_path))
        timer_test = utility.timer()
        model.load(model_path)
        T = {cls : [] for cls in class_list_wo_bg}
        P = {cls : [] for cls in class_list_wo_bg}
        iou_result = 0
        progbar = generic_utils.Progbar(len(test_imgs))
        for idx, img_data in enumerate(test_imgs):
            filepath = img_data['filepath']
            img = cv2.imread(filepath)
            X, fx, fy = format_img(img, args)
            X = np.transpose(X, (0, 2, 3, 1))

            all_dets = model.predict(X)

            t, p, iou = utility.get_map(all_dets, img_data['bboxes'], (fx, fy))
            iou_result += iou
            for key in t.keys():
                T[key].extend(t[key])
                P[key].extend(p[key])
            progbar.update(idx+1)
        all_aps = []
        for key in T.keys():
            ap = average_precision_score(T[key], P[key]) if T[key] else 0
            #print('{} AP: {}'.format(key, ap))
            if(math.isnan(ap)):ap = 0
            all_aps.append(ap)
        #mAP = np.mean(np.array(all_aps))
        iou_avg = iou_result/len(test_imgs)
        log_manager.add(all_aps, 'ap')
        log_manager.add(iou_avg, 'iou')
        log_manager.save()

        all_ap_dict = {cls:ap for cls, ap in zip(class_list_wo_bg, all_aps)}
        log_manager.write_cur_time()
        log_manager.write_log('model : {}\ntest : {}'.format(model_path, args.val_path)) 
        log_manager.write_log('Evaluation:')
        log_manager.write_log('iou: {:.3f}'.format(iou_avg))
        log_manager.write_log(
            'ap : {}\nmAP: {:.3f} (Best: {:.3f} @epoch {})'.format(
                str(all_ap_dict),
                np.mean(np.array(all_aps)),
                log_manager.best_map,
                log_manager.best_map_epoch
            )
        )
        log_manager.write_log('Runtime: {:.2f}s\n'.format(timer_test.toc()))


        '''
        print('mAP = {}'.format(np.mean(np.array(all_aps))))
        print('IoU@0.50 = ' + str(iou_result/len(test_imgs)))
        total_time = time.time() - begin
        print('Completely Elapsed time = {}'.format(total_time))
        print('Completely Elapsed time Per an image = {}'.format(total_time/len(test_imgs)))       
        '''
    

if __name__ == '__main__' :
    parser = Parser(args)
    model = Model(args)

    if(args.mode == 'train'):
        train(args, parser, model, log_manager)
    elif(args.mode == 'val'):
        val(args, parser, model)
    elif(args.mode == 'test'):
        test(args, parser, model)

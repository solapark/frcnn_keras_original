from option import args
from parser import Parser
from model import Model
import utility
from dataloader import DATALOADER

from keras.utils import generic_utils

import os
import cv2
import numpy as np

'''
def train(args, model, log_manager, img_preprocessor, train_dataloader, val_dataloader) :
    save_path = utility.make_save_dir(args)
    model_path_manager = utility.Model_path_manager(args) 

    log_manager.write_cur_time()
    log_manager.write('Training...')
    log_manager.write('Num train samples : len(train_dataloader) * num_cams',  len(train_dataloader), '*', args.num_cams)


    losses = np.zeros((args.epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = utility.timer()

    best_loss = np.Inf
    iter_num = 0
    for epoch_num in range(args.num_epochs):
        progbar = generic_utils.Progbar(args.epoch_length)
        log_manager.write('Epoch %d/%d'%(epoch_num+1, args.num_epochs))

        for idx in range(len(train_dataloader)):
            imgs_batch, labels_batch = dataloader[idx]
            X = img_preprocessor.process_batch(imgs_batch)
            loss = model_rpn.train(X, labels_batch)

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]


            progbar.update(iter_num+1, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),

            if (idx + 1) % self.args.print_every == 0:
                progbar.update(iter_num+1, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),

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
        mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
        rpn_accuracy_rpn_monitor = []
        print(f'Average number of overlapping bounding boxes from RPN = {mean_overlapping_bboxes} for {epoch_length} previous iterations')
        if mean_overlapping_bboxes == 0:
            print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')


    print('Training complete, exiting.')
'''

def calc_map(args, model, log_manager, img_preprocessor, dataloader):
    map_calculator = utility.Map_calculator(args)
    timer_test = utility.timer()
    progbar = generic_utils.Progbar(len(dataloader))
    for idx in range(len(dataloader)):
        #if(idx%40 !=0) : continue
        imgs_batch, labels_batch = dataloader[idx]
        gt_batch = map_calculator.get_gt_batch(labels_batch)
        X = img_preprocessor.process_batch(imgs_batch)
        if args.mv:
            all_dets = model.predict(X)
            for cam_idx in range(args.num_cam) : 
                dets = all_dets[cam_idx]
                gt = gt_batch[0][cam_idx]
                map_calculator.add_img_tp(dets, gt) 
        else: 
            for cam_idx in range(args.num_cam):
                x = X[cam_idx]
                all_dets = model.predict(x)

                gt = gt_batch[0][cam_idx]
                map_calculator.add_img_tp(all_dets, gt) 

        progbar.update(idx+1)
    
    all_aps = map_calculator.get_aps()
    iou_avg = map_calculator.get_iou()

    log_manager.add(all_aps, 'ap')
    log_manager.add(iou_avg, 'iou')
    log_manager.save()

    all_ap_dict = map_calculator.get_aps_dict()
    log_manager.write_cur_time()
    log_manager.write_log('Evaluation:')
    log_manager.write_log('iou: {:.3f}'.format(iou_avg))
    log_manager.write_log(
        'ap : {}\nmAP: {:.3f} (Best: {:.3f} @epoch {})'.format(
            str(all_ap_dict),
            map_calculator.get_map(),
            log_manager.best_map,
            log_manager.best_map_epoch
        )
    )
    log_manager.write_log('Runtime: {:.2f}s\n'.format(timer_test.toc()))


def val_models(args, model, log_manager, img_preprocessor, val_dataloader):
    model_path_manager = utility.Model_path_manager(args)
    
    #all_model_path = model_path_manager.get_all_path()
    all_model_path = model_path_manager.get_all_path()[-2:-1]
    for i, model_path in enumerate(all_model_path) :
        model.load(model_path)
        log_manager.write_log('model : {}\n'.format(model_path)) 
        calc_map(args, model, log_manager, img_preprocessor, val_dataloader)   

if __name__ == '__main__' :
    model = Model(args)
    log_manager = utility.Log_manager(args)
    img_preprocessor = utility.Img_preprocessor(args)

    if(args.mode == 'train'):
        train_dataloader = DATALOADER(args, 'train', args.train_path)
        val_dataloader = DATALOADER(args, 'val', args.val_path)
        train(args, model, log_manager, img_preprocessor, train_dataloader, val_dataloader)
    elif(args.mode == 'val'):
        val_dataloader = DATALOADER(args, 'val', args.val_path)
        calc_map(args,model, log_manager, img_preprocessor, val_dataloader)
    elif(args.mode == 'test'):
        test(args, model, log_manager)
    elif(args.mode == 'val_models'):
        val_dataloader = DATALOADER(args, 'val', args.val_models_path)
        val_models(args,model, log_manager, img_preprocessor, val_dataloader)

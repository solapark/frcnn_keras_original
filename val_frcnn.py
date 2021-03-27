from option import args
from model import Model
import utility
from dataloader import DATALOADER
from gt.rpn_gt_calculator import RPN_GT_CALCULATOR

from keras.utils import generic_utils

import os
import cv2
import numpy as np

def train(args, model, log_manager, img_preprocessor, train_dataloader, val_dataloader) :
    model_path_manager = utility.Model_path_manager(args)
    sv_gt_batch_generator = utility.Sv_gt_batch_generator(args)
    rpn_gt_calculator = RPN_GT_CALCULATOR(args)

    log_manager.write_cur_time()
    log_manager.write_log('Training...')
    log_manager.write_log('Num train samples : len(train_dataloader) * num_cam ='+str(len(train_dataloader))+'*'+str(args.num_cam))

    timer_data, timer_model = utility.timer(), utility.timer()
    best_mAP = 0
    for epoch_num in range(args.num_epochs):
        log_manager.write_log('[Epoch %d/%d]'%(epoch_num+1, args.num_epochs))

        for idx in range(len(train_dataloader)):
        #for idx in range(10):
            timer_data.tic()
            X, Y = train_dataloader[idx]
            X = img_preprocessor.process_batch(X)
            rpn_gt_batch = rpn_gt_calculator.get_batch(Y)
            timer_data.hold()

            timer_model.tic()
            if args.mv :
                loss, num_calssifier_pos_samples = model.train_batch(X, Y, rpn_gt_batch)
            else :
                Y = sv_gt_batch_generator.get_gt_batch(Y)
                loss_list = []
                num_calssifier_pos_samples_list = []
                timer_model.tic()
                for cam_idx in range(args.num_cam):
                    x = X[cam_idx]
                    y = Y[0][cam_idx]
                    cur_rpn_gt_batch = rpn_gt_batch[cam_idx*2:cam_idx*2+2]

                    loss, num_calssifier_pos_samples = model.train_batch(x, y, cur_rpn_gt_batch)
                    loss_list.append(loss)
                    num_calssifier_pos_samples_list.append(num_calssifier_pos_samples)
                loss = np.array(loss_list).mean(0)
                num_calssifier_pos_samples = np.array(num_calssifier_pos_samples_list).mean()

            timer_model.hold()
            log_manager.add(loss, 'loss')
            log_manager.add(num_calssifier_pos_samples, 'num_calssifier_pos_samples')

            if (idx + 1) % args.print_every == 0:
                log_manager.write_log('[%d/%d]\t[%s]\t[%s]\t%.2fs + %.2fs'%(
                    idx+1, 
                    len(train_dataloader), 
                    log_manager.display('loss'), 
                    log_manager.display('num_calssifier_pos_samples'), 
                    timer_data.release(), 
                    timer_model.release()))
            timer_data.tic()
        log_manager.epoch_done()
    
        cur_mAP = calc_map(args, model, log_manager, img_preprocessor, val_dataloader)
        if(cur_mAP > best_mAP):
            best_mAP = cur_mAP
            model.save(model_path_manager.get_path('best'))
        model.save(model_path_manager.get_path('last'))

def calc_map(args, model, log_manager, img_preprocessor, dataloader):
    sv_gt_batch_generator = utility.Sv_gt_batch_generator(args)
    map_calculator = utility.Map_calculator(args)
    timer_test = utility.timer()
    progbar = generic_utils.Progbar(len(dataloader))
    for idx in range(len(dataloader)):
    #for idx in range(3):
        #if(idx%40 !=0) : continue
        imgs_batch, labels_batch = dataloader[idx]
        gt_batch = sv_gt_batch_generator.get_gt_batch(labels_batch)
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
    cur_map = map_calculator.get_map()
    log_manager.write_cur_time()
    log_manager.write_log('Evaluation:')
    log_manager.write_log('iou: {:.3f}'.format(iou_avg))
    log_manager.write_log(
        'ap : {}\nmAP: {:.3f} (Best: {:.3f} @epoch {})'.format(
            str(all_ap_dict),
            cur_map,
            log_manager.best_map,
            log_manager.best_map_epoch
        )
    )
    log_manager.write_log('Runtime: {:.2f}s\n'.format(timer_test.toc()))
    return cur_map

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
    utility.file_system(args)
    log_manager = utility.Log_manager(args)
    img_preprocessor = utility.Img_preprocessor(args)

    if(args.mode == 'train'):
        train_dataloader = DATALOADER(args, 'train', args.train_path)
        val_dataloader = DATALOADER(args, 'val', args.val_path)
        #model.load(args.input_weight_path)
        train(args, model, log_manager, img_preprocessor, train_dataloader, val_dataloader)
    elif(args.mode == 'val'):
        val_dataloader = DATALOADER(args, 'val', args.val_path)
        #model.load(args.model_load_path)
        calc_map(args,model, log_manager, img_preprocessor, val_dataloader)
    elif(args.mode == 'test'):
        test(args, model, log_manager)
    elif(args.mode == 'val_models'):
        val_dataloader = DATALOADER(args, 'val', args.val_models_path)
        val_models(args,model, log_manager, img_preprocessor, val_dataloader)

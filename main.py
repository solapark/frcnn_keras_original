from option import args
from model import Model
import utility
from dataloader import DATALOADER

from keras.utils import generic_utils

import os
import cv2
import numpy as np
import math

def train(args, model, log_manager, img_preprocessor, train_dataloader, val_dataloader) :
    model_path_manager = utility.Model_path_manager(args)

    log_manager.write_cur_time()
    log_manager.write_log('Training...')
    log_manager.write_log('Num train samples : len(train_dataloader) * num_valid_cam ='+str(len(train_dataloader))+'*'+str(args.num_valid_cam))

    timer_data, timer_model = utility.timer(), utility.timer()
    start_epoch = model_path_manager.get_resume_epoch() + 1 if args.resume else 1
    for epoch_num in range(start_epoch, args.num_epochs+1):
        log_manager.write_log('[Epoch %d/%d]'%(epoch_num, args.num_epochs))

        for idx in range(len(train_dataloader)):
        #for idx in range(10):
            timer_data.tic()
            extirns = None
            rpn_results = None
            if args.is_use_epipolar and args.freeze_rpn :
                X_raw, Y, extrins, rpn_results = train_dataloader[idx]
            elif args.is_use_epipolar :
                X_raw, Y, extrins = train_dataloader[idx]
            else : 
                X_raw, Y = train_dataloader[idx]
            X = img_preprocessor.process_batch(X_raw)
            timer_data.hold()

            timer_model.tic()

            loss, num_calssifier_pos_samples = model.train_batch(X, Y, X_raw, extrins, rpn_results)

            timer_model.hold()

            print('loss', loss) 
            #print('num_calssifier_pos_samples', num_calssifier_pos_samples)

            if(loss[-1] == np.inf) : continue 

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
    
        model.save(model_path_manager.get_path('%d'%(epoch_num)))

def calc_map(args, model, log_manager, img_preprocessor, dataloader):
    map_calculator = utility.Map_calculator(args)
    sv_gt_batch_generator = utility.Sv_gt_batch_generator(args)
    timer_test = utility.timer()
    progbar = generic_utils.Progbar(len(dataloader))
    for idx in range(len(dataloader)):
    #for idx in range(0, len(dataloader), 4):
        extirns = None
        if args.is_use_epipolar :
            X_raw, Y, extrins = dataloader[idx]
        else : 
            X_raw, Y = dataloader[idx]
        X = img_preprocessor.process_batch(X_raw)

        all_dets = model.predict_batch(X, extrins)


        gt_batch = sv_gt_batch_generator.get_gt_batch(Y)

        for cam_idx in range(args.num_valid_cam) : 
            dets = all_dets[cam_idx]
            gt = gt_batch[0][cam_idx]
            map_calculator.add_img_tp(dets, gt) 

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
    
    all_model_path = model_path_manager.get_path_in_range(args.val_start_idx, args.val_end_idx, args.val_interval)

    for i, model_path in enumerate(all_model_path) :
        if not os.path.isfile(model_path) :
            log_manager.write_log('{} doesn\'t exist\n'.format(model_path)) 
            break
        #model.load(model_path)
        log_manager.write_log('model : {}\n'.format(model_path)) 
        calc_map(args, model, log_manager, img_preprocessor, val_dataloader)   

def demo(args, model, log_manager, img_preprocessor, dataloader):
    progbar = generic_utils.Progbar(len(dataloader))
    result_saver = utility.Result_saver(args)
    #model.load(args.input_weight_path)
    for idx in range(len(dataloader)):
        extrins = None
        if args.is_use_epipolar :
            X_raw, Y, img_paths, extrins = dataloader[idx]
        else : 
            X_raw, Y, img_paths = dataloader[idx]

        X = img_preprocessor.process_batch(X_raw)
        #all_dets = []
        #result_saver.save(X_raw, Y, img_paths, all_dets)
        all_dets = model.predict_batch(X, extrins, X_raw)
        result_saver.save(X_raw, Y, img_paths, all_dets)
        progbar.update(idx+1)

def save_rpn_feature(args, model, log_manager, img_preprocessor, dataloader) :
    rpn_result_saver = utility.Rpn_result_saver(args)
    progbar = generic_utils.Progbar(len(dataloader))

    for idx in range(len(dataloader)):
        X_raw, img_paths = dataloader[idx]
        X = img_preprocessor.process_batch(X_raw)

        rpn_result = model.rpn_predict_batch(X, X_raw)

        rpn_result_saver.save(img_paths, rpn_result)
        progbar.update(idx+1)

if __name__ == '__main__' :
    model = Model(args)
    #model = None
    utility.file_system(args)
    log_manager = utility.Log_manager(args)
    img_preprocessor = utility.Img_preprocessor(args)

    if(args.mode == 'train'):
        train_dataloader = DATALOADER(args, 'train', args.train_path)
        train(args, model, log_manager, img_preprocessor, train_dataloader, None)
    elif(args.mode == 'val'):
        val_dataloader = DATALOADER(args, 'val', args.val_path)
        calc_map(args,model, log_manager, img_preprocessor, val_dataloader)
    elif(args.mode == 'val_models'):
        val_dataloader = DATALOADER(args, 'val', args.val_path)
        val_models(args,model, log_manager, img_preprocessor, val_dataloader)
    elif(args.mode == 'demo'):
        dataloader = DATALOADER(args, 'demo', args.demo_path)
        demo(args, model, log_manager, img_preprocessor, dataloader)
    elif(args.mode == 'save_rpn_feature'):
        dataloader = DATALOADER(args, 'save_rpn_feature', args.train_path)
        save_rpn_feature(args, model, log_manager, img_preprocessor, dataloader)

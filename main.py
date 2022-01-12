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
        #for idx in range(100):
            timer_data.tic()
            images, labels, image_paths, extrins, rpn_results, ven_results = train_dataloader[idx]
            X = img_preprocessor.process_batch(images)
            timer_data.hold()

            timer_model.tic()
            loss, num_calssifier_pos_samples = model.train_batch(X, images, labels, image_paths, extrins, rpn_results, ven_results)
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

    dataset_interval = 18 if args.fast_val else 1
    dataset_size = len(dataloader) // dataset_interval
    progbar = generic_utils.Progbar(dataset_size)
    for idx in range(0, len(dataloader), dataset_interval):
        images, labels, image_paths, extrins, rpn_results, ven_results = dataloader[idx]
        X = img_preprocessor.process_batch(images)
        all_dets = model.predict_batch(X, images, extrins, rpn_results, ven_results)

        gt_batch = sv_gt_batch_generator.get_gt_batch(labels)
 
        for cam_idx in range(args.num_valid_cam) : 
            dets = all_dets[cam_idx]
            gt = gt_batch[0][cam_idx]
            map_calculator.add_img_tp(dets, gt) 

        progbar.update(idx/dataset_interval+1)
    
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
    for cls, prob in all_ap_dict.items():
        log_manager.write_log('%s\t%.4f'%(cls, prob))
    log_manager.write_log('mAP\t%.4f'%(cur_map))
    log_manager.write_log('Runtime: {:.2f}s\n'.format(timer_test.toc()))
    return cur_map

def val_models(args, model, log_manager, img_preprocessor, val_dataloader):
    model_path_manager = utility.Model_path_manager(args)
    
    all_model_path = model_path_manager.get_path_in_range(args.val_start_idx, args.val_end_idx, args.val_interval)

    for i, model_path in enumerate(all_model_path) :
        if not os.path.isfile(model_path) :
            log_manager.write_log('{} doesn\'t exist\n'.format(model_path)) 
            break
        model.load(model_path)
        log_manager.write_log('model : {}\n'.format(model_path)) 
        calc_map(args, model, log_manager, img_preprocessor, val_dataloader)   

def demo(args, model, img_preprocessor, dataloader):
    progbar = generic_utils.Progbar(len(dataloader))
    result_saver = utility.Result_saver(args)
    #model.load(args.input_weight_path)
    for idx in range(len(dataloader)):
        images, labels, image_paths, extrins, rpn_results, ven_results = dataloader[idx]
        X = img_preprocessor.process_batch(images)
        all_dets = model.predict_batch(X, images, extrins, rpn_results, ven_results)
        result_saver.save(images, image_paths, all_dets)
        progbar.update(idx+1)

def draw_json(args, dataloader):
    progbar = generic_utils.Progbar(len(dataloader))
    result_saver = utility.Result_saver(args)
    if 'sv' in args.dataset_path.split('/')[-1] and 'mvcnn' not in args.dataset_path.split('/')[-1]:
        num2cls_with_bg = args.sv_num2cls_with_bg 
    else :
        num2cls_with_bg = args.num2cls_with_bg 

    for idx in range(len(dataloader)):
        images, labels, image_paths, extrins, rpn_results, ven_results = dataloader[idx]
        
        #new_labels = utility.labels_to_draw_format(labels, args.num_cam, num2cls_with_bg, args.resize_ratio)
        new_labels = utility.labels_to_draw_format(labels, args.num_cam, num2cls_with_bg)
        result_saver.save(images, image_paths, new_labels)
        progbar.update(idx+1)

def write_json(args, model, img_preprocessor, dataloader):
    progbar = generic_utils.Progbar(len(dataloader))
    json_writer = utility.Json_writer(args)
    for idx in range(len(dataloader)):
        images, labels, image_paths, extrins, rpn_results, ven_results = dataloader[idx]
        X = img_preprocessor.process_batch(images)
        all_dets = model.predict_batch(X, images, extrins, rpn_results, ven_results)
        json_writer.write(all_dets, image_paths)
        progbar.update(idx+1)

    json_writer.close()

def val_json_json(args, gt_dataloader, pred_dataloader):
    utility.file_system(args)
    log_manager = utility.Log_manager(args)
    map_calculator = utility.Map_calculator(args)
    sv_gt_batch_generator = utility.Sv_gt_batch_generator(args)
    timer_test = utility.timer()

    if 'sv' in args.pred_dataset_path.split('/')[-1] and 'mvcnn' not in args.pred_dataset_path.split('/')[-1]:
        num2cls_with_bg = args.sv_num2cls_with_bg 
    else :
        num2cls_with_bg = args.num2cls_with_bg 


    progbar = generic_utils.Progbar(len(gt_dataloader))
    for idx in range(len(gt_dataloader)):
        _, gt_labels, _, _, _, _  = gt_dataloader[idx]
        _, pred_labels, _, _, _, _  = pred_dataloader[idx]

        gt_batch = sv_gt_batch_generator.get_gt_batch(gt_labels)
        all_dets = utility.labels_to_draw_format(pred_labels, args.num_cam, num2cls_with_bg)
 
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
    for cls, prob in all_ap_dict.items():
        log_manager.write_log('%s\t%.4f'%(cls, prob))
    log_manager.write_log('mAP\t%.4f'%(cur_map))
    log_manager.write_log('Runtime: {:.2f}s\n'.format(timer_test.toc()))
    return cur_map


def save_rpn_feature(args, model, img_preprocessor, dataloader) :
    rpn_result_saver = utility.Pickle_result_saver(args, args.rpn_pickle_dir)
    progbar = generic_utils.Progbar(len(dataloader))

    for idx in range(len(dataloader)):
        X_raw, _, img_paths, _, _, _ = dataloader[idx]
        X = img_preprocessor.process_batch(X_raw)

        rpn_result = model.rpn_predict_batch(X, X_raw)

        rpn_result_saver.save(img_paths[0], rpn_result)
        progbar.update(idx+1)

def save_ven_feature(args, model, img_preprocessor, dataloader) :
    ven_result_saver = utility.Pickle_result_saver(args, args.ven_pickle_dir)
    progbar = generic_utils.Progbar(len(dataloader))

    for idx in range(len(dataloader)):
        X_raw, _, img_paths, extrins, rpn_results, _ = dataloader[idx]
        X = img_preprocessor.process_batch(X_raw)

        ven_result = model.ven_predict_batch(X, X_raw, extrins, rpn_results)

        ven_result_saver.save(img_paths[0], ven_result)
        progbar.update(idx+1)

def save_sv_wgt(args, model):
    model.load_sv_wgt(args.input_weight_path)
    model.save(args.output_weight_path)

if __name__ == '__main__' :
    model = Model(args)
    #model = None
    img_preprocessor = utility.Img_preprocessor(args)

    if(args.mode == 'train'):
        utility.file_system(args)
        log_manager = utility.Log_manager(args)
        train_dataloader = DATALOADER(args, 'train', args.dataset_path)
        train(args, model, log_manager, img_preprocessor, train_dataloader, None)
    elif(args.mode == 'val'):
        log_manager = utility.Log_manager(args)
        val_dataloader = DATALOADER(args, 'val', args.dataset_path)
        calc_map(args,model, log_manager, img_preprocessor, val_dataloader)
    elif(args.mode == 'val_models'):
        log_manager = utility.Log_manager(args)
        val_dataloader = DATALOADER(args, 'val', args.dataset_path)
        val_models(args,model, log_manager, img_preprocessor, val_dataloader)
    elif(args.mode == 'demo'):
        dataloader = DATALOADER(args, 'demo', args.dataset_path)
        demo(args, model, img_preprocessor, dataloader)
    elif(args.mode == 'save_rpn_feature'):
        dataloader = DATALOADER(args, 'save_rpn_feature', args.dataset_path)
        save_rpn_feature(args, model, img_preprocessor, dataloader)
    elif(args.mode == 'save_ven_feature'):
        dataloader = DATALOADER(args, 'save_ven_feature', args.dataset_path)
        save_ven_feature(args, model, img_preprocessor, dataloader)
    elif(args.mode == 'save_sv_wgt'):
        save_sv_wgt(args, model)

    elif(args.mode == 'draw_json'):
        dataloader = DATALOADER(args, 'draw_json', args.dataset_path)
        draw_json(args, dataloader)

    elif(args.mode == 'val_json_json'):
        gt_dataloader = DATALOADER(args, 'val', args.dataset_path)
        pred_dataloader = DATALOADER(args, 'val', args.pred_dataset_path)
        val_json_json(args, gt_dataloader, pred_dataloader)

    elif(args.mode == 'write_json'):
        dataloader = DATALOADER(args, 'demo', args.dataset_path)
        write_json(args, model, img_preprocessor, dataloader)



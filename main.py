from option import args
from model import Model
import evaluation
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

            #print('loss', loss) 
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
    
        if epoch_num % args.save_interval ==0 :
            model.save(model_path_manager.get_path('%d'%(epoch_num)))

def calc_map(args, model, log_manager, img_preprocessor, dataloader):
    map_calculator = evaluation.Map_calculator(args)
    sv_gt_batch_generator = utility.Sv_gt_batch_generator(args)
    det_resizer = utility.Det_resizer(args)
    timer_test = utility.timer()

    dataset_interval = 18 if args.fast_val else 1
    dataset_size = len(dataloader) // dataset_interval
    progbar = generic_utils.Progbar(dataset_size)
    for idx in range(0, len(dataloader), dataset_interval):
        images, labels, image_paths, extrins, rpn_results, ven_results = dataloader[idx]
        X = img_preprocessor.process_batch(images)
        all_dets = model.predict_batch(X, images, extrins, rpn_results, ven_results)
        if args.mode != 'val_models' : 
            det_resizer.resize(all_dets)

        gt_batch = sv_gt_batch_generator.get_gt_batch(labels)
 
        for cam_idx in range(args.num_valid_cam) : 
            dets = all_dets[cam_idx]
            gt = gt_batch[0][cam_idx]
            map_calculator.add_tp_fp(dets, gt) 

        progbar.update(idx/dataset_interval+1)
    
    evaluation.log_eval(map_calculator, log_manager)

    '''
    all_aps = map_calculator.get_aps()
    #iou_avg = map_calculator.get_iou()

    log_manager.add(all_aps, 'ap')
    #log_manager.add(iou_avg, 'iou')
    log_manager.save()

    all_ap_dict = map_calculator.get_aps_dict()
    cur_map = map_calculator.get_map()
    log_manager.write_cur_time()
    log_manager.write_log('Evaluation:')
    log_manager.write_log('mAP\t%.2f'%(cur_map*100))
    #log_manager.write_log('iou\t{:.3f}'.format(iou_avg))
    log_manager.write_log('Runtime(s)\t{:.2f}'.format(timer_test.toc()))
    for i, (cls, prob) in enumerate(all_ap_dict.items()):
        if prob<0 : continue
        log_manager.write_log('%s\t%.2f'%(cls, prob*100))
    log_manager.write_log('\n')
    '''

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
    L2D = utility.Labels_to_draw_format(args)

    for idx in range(len(dataloader)):
        images, labels, image_paths, extrins, rpn_results, ven_results = dataloader[idx]
        new_labels = L2D.labels_to_draw_format(labels)
        if args.json_nms : 
            new_labels = utility.json_nms(new_labels, args)
        result_saver.save(images, image_paths, new_labels)
        progbar.update(idx+1)

def write_json(args, model, img_preprocessor, dataloader):
    progbar = generic_utils.Progbar(len(dataloader))
    json_writer = utility.Json_writer(args)
    rp_to_write_format = utility.RP_to_write_format(args)
    reid_to_write_format = utility.Reid_to_write_format(args)
    for idx in range(len(dataloader)):
        images, labels, image_paths, extrins, rpn_results, ven_results = dataloader[idx]
        X = img_preprocessor.process_batch(images)
        if args.write_rpn_only :
            pred_box_batch, pred_box_prob_batch = model.predict_rpn_only_batch(X, images, extrins, rpn_results, ven_results)
            all_dets = rp_to_write_format.rp_to_write_format(pred_box_batch, pred_box_prob_batch)
        elif args.write_reid :
            reid_box_pred_batch, is_valid_batch = model.predict_reid_batch(X, images, extrins, rpn_results, ven_results)
            all_dets = reid_to_write_format.reid_to_write_format(reid_box_pred_batch, is_valid_batch)

        else :
            all_dets = model.predict_batch(X, images, extrins, rpn_results, ven_results)

        json_writer.write(all_dets, image_paths)
        progbar.update(idx+1)
        #if(idx==50) : break

    json_writer.close()

def comp_json(args, gt_dataloader, pred_dataloader1, pred_dataloader2):
    #num_test = 100
    utility.file_system(args)
    log_manager = utility.Log_manager(args)
    map_calculator1 = evaluation.Map_calculator(args)
    map_calculator2 = evaluation.Map_calculator(args)
    sv_gt_batch_generator = utility.Sv_gt_batch_generator(args)
    L2D = utility.Labels_to_draw_format(args)
    timer_test = utility.timer()

    progbar = generic_utils.Progbar(len(gt_dataloader))
    data_len = min(len(pred_dataloader1), len(pred_dataloader2))
    for idx in range(data_len):
        _, gt_labels, path, _, _, _  = gt_dataloader[idx]
        _, pred_labels1, _, _, _, _  = pred_dataloader1[idx]
        _, pred_labels2, _, _, _, _  = pred_dataloader2[idx]

        gt_batch = sv_gt_batch_generator.get_gt_batch(gt_labels)
        all_dets1 = L2D.labels_to_draw_format(pred_labels1)
        all_dets2 = L2D.labels_to_draw_format(pred_labels2)
 
        for cam_idx in range(args.num_valid_cam) : 
            log_manager.write_log('path : %s'%(path[0][int(cam_idx)+1]))

            dets1 = all_dets1[cam_idx]
            dets2 = all_dets2[cam_idx]
            gt = gt_batch[0][cam_idx]

            map_calculator1.reset()
            map_calculator2.reset()

            map_calculator1.add_tp_fp(dets1, gt) 
            map_calculator2.add_tp_fp(dets2, gt) 

            eval1 = map_calculator1.get_eval()
            eval2 = map_calculator2.get_eval()

            mean_eval1 = map_calculator1.get_mean_eval()
            mean_eval2 = map_calculator2.get_mean_eval()

            metric = map_calculator1.metric

            for metric_name, m_ev1, m_ev2 in zip(metric, mean_eval1, mean_eval2) :
                log_manager.write_log('%s\t%.2f\t%.2f'%(metric_name, m_ev1, m_ev2))
            log_manager.write_log('\n')

            valid_cls = set(eval1[0].keys()).union(set(eval2[0].keys()))
            valid_cls = sorted(list(valid_cls))

            for metric_name, ev1, ev2, m_ev1, m_ev2 in zip(metric, eval1, eval2, mean_eval1, mean_eval2) :
                log_manager.write_log('%s\t%.2f\t%.2f'%(metric_name, m_ev1, m_ev2))
                log_manager.write_log('\n')

                for i, cls in enumerate(valid_cls):
                    e1 = ev1[cls]
                    e2 = ev2[cls]

                    if e1<0 and e2 < 0: continue

                    if e1 > e2 :
                        log_manager.write_log('*%s\t\t%.2f\t%.2f'%(cls, e1, e2))
                    else :
                        log_manager.write_log('%s\t\t%.2f\t%.2f'%(cls, e1, e2))

                    log_manager.write_log('tp\t\t%d\t%d'%(sum(map_calculator1.TP[cls]), sum(map_calculator2.TP[cls])))
                    log_manager.write_log('fp\t\t%d\t%d'%(sum(map_calculator1.FP[cls]), sum(map_calculator2.FP[cls])))
                    log_manager.write_log('gt\t\t%d\t%d'%(map_calculator1.gt_counter_per_class[cls], map_calculator2.gt_counter_per_class[cls]))

                log_manager.write_log('\n')
 
            '''
            all_aps1 = map_calculator1.get_aps()
            all_aps2 = map_calculator2.get_aps()
            #iou_avg = map_calculator.get_iou()

            log_manager.add(all_aps1, 'ap1')
            log_manager.add(all_aps2, 'ap2')
            #log_manager.add(iou_avg, 'iou')
            log_manager.save()

            all_ap_dict1 = map_calculator1.get_aps_dict()
            cur_map1 = map_calculator1.get_map()

            all_ap_dict2 = map_calculator2.get_aps_dict()
            cur_map2 = map_calculator2.get_map()

            log_manager.write_log('mAP1\t%.2f\tmAP2\t%.2f'%(cur_map1*100, cur_map2*100))
            log_manager.write_log('\n')

            valid_cls = set(all_ap_dict1.keys()).union(set(all_ap_dict2.keys()))
            valid_cls = sorted(list(valid_cls))
            log_manager.write_log('cls\t\tprob1\tprob2')
            for i, cls in enumerate(valid_cls):
                prob1 = all_ap_dict1[cls]
                prob2 = all_ap_dict2[cls]
                if prob1<0 and prob2 < 0: continue
                if prob1 > prob2 :
                    log_manager.write_log('*%s\t\t%.2f\t%.2f'%(cls, prob1*100, prob2*100))
                else :
                    log_manager.write_log('%s\t\t%.2f\t%.2f'%(cls, prob1*100, prob2*100))
            log_manager.write_log('\n')
            '''

        progbar.update(idx+1)

    return cur_map1, cur_map2

def val_json_json(args, gt_dataloader, pred_dataloader):
    #num_test = 100
    utility.file_system(args)
    log_manager = utility.Log_manager(args)
    map_calculator = evaluation.Map_calculator(args)
    sv_gt_batch_generator = utility.Sv_gt_batch_generator(args)
    L2D = utility.Labels_to_draw_format(args)
    timer_test = utility.timer()

    progbar = generic_utils.Progbar(len(gt_dataloader))
    #for idx in range(len(gt_dataloader)):
    for idx in range(len(pred_dataloader)):
        _, gt_labels, path, _, _, _  = gt_dataloader[idx]
        _, pred_labels, _, _, _, _  = pred_dataloader[idx]

        gt_batch = sv_gt_batch_generator.get_gt_batch(gt_labels)
        all_dets = L2D.labels_to_draw_format(pred_labels)
 
        if args.json_nms : 
            all_dets = utility.json_nms(all_dets, args)

        for cam_idx in range(args.num_valid_cam) : 
            dets = all_dets[cam_idx]
            gt = gt_batch[0][cam_idx]
            map_calculator.add_tp_fp(dets, gt) 

            #if idx*3+cam_idx+1 == num_test : break

        progbar.update(idx+1)
        #if idx == 10 : break

    evaluation.log_eval(map_calculator, log_manager)

    '''
    all_aps = map_calculator.get_aps()
    #iou_avg = map_calculator.get_iou()

    log_manager.add(all_aps, 'ap')
    #log_manager.add(iou_avg, 'iou')
    log_manager.save()

    all_ap_dict = map_calculator.get_aps_dict()
    cur_map = map_calculator.get_map()
    log_manager.write_cur_time()
    log_manager.write_log('Evaluation:')
    log_manager.write_log('mAP\t%.2f'%(cur_map*100))
    #log_manager.write_log('iou\t{:.3f}'.format(iou_avg))
    log_manager.write_log('Runtime(s)\t{:.2f}'.format(timer_test.toc()))
    '''

    '''
    for i, (cls, v) in enumerate(map_calculator.gt_counter_per_class.items()):
        if v<1 : continue
        log_manager.write_log('%03d\t%d'%(i, v))
    log_manager.write_log('\n')

    for i, (cls, v) in enumerate(map_calculator.gt_counter_per_class.items()):
        tp = int(sum(map_calculator.TP[cls]))
        fp = int(sum(map_calculator.FP[cls]))

        if tp == 0 and fp == 0: continue

        log_manager.write_log('%03d\t%d\t%d'%(i, tp, fp))
    log_manager.write_log('\n')
    '''

    '''
    for i, (cls, prob) in enumerate(all_ap_dict.items()):
        if prob<0 : continue
        log_manager.write_log('%s\t%.2f'%(cls, prob*100))
        #log_manager.write_log('%03d\t%.2f'%(i, prob*100))
    log_manager.write_log('\n')
    return cur_map
    '''


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

def save_reid_input(args, model, img_preprocessor, dataloader) :
    reid_input_result_saver = utility.Pickle_result_saver(args, args.reid_input_pickle_dir)
    progbar = generic_utils.Progbar(len(dataloader))

    for idx in range(len(dataloader)):
        images, labels, image_paths, extrins, rpn_results, ven_results = dataloader[idx]
        X = img_preprocessor.process_batch(images)

        reid_input = model.get_reid_input(X, images, labels, image_paths, extrins, rpn_results, ven_results)

        reid_input_result_saver.save(image_paths[0], reid_input)
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
        val_dataloader = DATALOADER(args, 'val_models', args.dataset_path)
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
    elif(args.mode == 'save_reid_input'):
        dataloader = DATALOADER(args, 'save_rpn_feature', args.dataset_path)
        save_reid_input(args, model, img_preprocessor, dataloader)
    elif(args.mode == 'save_sv_wgt'):
        save_sv_wgt(args, model)

    elif(args.mode == 'draw_json'):
        dataloader = DATALOADER(args, 'draw_json', args.dataset_path)
        draw_json(args, dataloader)

    elif(args.mode == 'val_json_json'):
        gt_dataloader = DATALOADER(args, 'val', args.dataset_path)
        pred_dataloader = DATALOADER(args, 'val', args.pred_dataset_path)
        val_json_json(args, gt_dataloader, pred_dataloader)

    elif(args.mode == 'comp_json'):
        gt_dataloader = DATALOADER(args, 'val', args.dataset_path)
        pred_dataloader1 = DATALOADER(args, 'val', args.pred_dataset_path1)
        pred_dataloader2 = DATALOADER(args, 'val', args.pred_dataset_path2)
        comp_json(args, gt_dataloader, pred_dataloader1, pred_dataloader2)

    elif(args.mode == 'write_json'):
        utility.file_system(args)
        dataloader = DATALOADER(args, 'demo', args.dataset_path)
        write_json(args, model, img_preprocessor, dataloader)



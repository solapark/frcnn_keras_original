from tqdm import tqdm

import utility, utils
from bipartite_graph import Bipartite_graph
from json_maker import json_maker

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import average_precision_score, precision_recall_curve

import argparse 
import sys
import numpy as np
import os

#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

def draw_precision_recall_curve(T, P, classes, save_dir) : 
    classes = classes[:-2]

    for i, cls in enumerate(classes) :
        precision, recall, threshold = precision_recall_curve(T[cls], P[cls])
        save_path = os.path.join(save_dir, '%03d_%s.png'%(i, cls))

        ap = 0
        for j in range(len(threshold)-1) : 
            ap+= precision[j+1] * (recall[j+1] - recall[j+2])

        fig = plt.figure(figsize = (9,6))
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.plot(recall, precision)
        plt.scatter(recall,precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('%s_%.2f'%(cls, ap))
        plt.savefig(save_path)
        plt.clf()

def print_list(name, l):
    btw = ','
    print(name, end = btw)
    print(btw.join(l))

def calc_map(T, P, classes) :
    classes = classes[:-2]
    #ap = [average_precision_score(T[cls], P[cls]) for cls in classes]
    #ap = np.array(ap)

    aps = []
    for cls in classes :
        precision, recall, threshold = precision_recall_curve(T[cls], P[cls])

        ap = 0
        for j in range(len(threshold)-1) : 
            ap+= precision[j+1] * (recall[j+1] - recall[j+2])
        aps.append(ap)
    ap = np.array(aps)

    mAP = np.round(np.mean(ap), 3)
    ap = np.round(ap, 3).astype('str').tolist()

    print('map,%f'%(mAP))
    print_list('classes', classes)
    print_list('ap', ap)


def print_TF (cm, classes):
    TP = cm.diagonal()
    FP = np.sum(cm, 0) - TP
    FN = cm[:, -1]

    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    F1 = 2/(1/precision + 1/recall)

    precision = np.round(precision, 2)
    recall = np.round(recall, 2)
    F1 = np.round(F1, 2)

    classes = classes[:-2]
    TP = TP[:-2].astype('str').tolist()
    FP = FP[:-2].astype('str').tolist()
    FN = FN[:-2].astype('str').tolist()
    precision = precision[:-2].astype('str').tolist()
    recall = recall[:-2].astype('str').tolist()
    F1 = F1[:-2].astype('str').tolist()

    print_list('class', classes)
    print_list('TP', TP)
    print_list('FP', FP)
    print_list('FN', FN)
    print_list('precision', precision)
    print_list('recall', recall)
    print_list('F1', F1)

def plot_confusion_matrix(cm, y_true, y_pred, gt_classes, pred_classes, save_path, is_text,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Confusion matrix, with normalization'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    #print(cm)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #cm[np.argmax(cm, axis=1)] = 0
        #cm += .5 
        #cm = np.clip(cm, 0, 1)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize= (40, 40))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=pred_classes, yticklabels=gt_classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if is_text :
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = format(cm[i, j], fmt)
                if float(value) :
                    ax.text(j, i, value,
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()

    #plt.rcParams["figure.figsize"] = (10000, 10000)
    plt.savefig(save_path)
    plt.show()

    return ax


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, default='/data1/sap/MessyTable/labels/test.json')
    parser.add_argument('--src_path', type=str, default='/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log.json')
    parser.add_argument('--thresh', type=float, default=.0)
    parser.add_argument('--target_class', type=str, default=None)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--text', action='store_true')

    parser.add_argument('--num_cam', type=int, default=3)
    parser.add_argument('--class_names_dict_path', type=str, default='/data1/sap/MessyTable/class.dict')
    parser.add_argument('--save_dir', type=str, default='/data3/sap/frcnn_keras_original/experiment')

    args = parser.parse_args()

    names_dict = utils.get_dict_from_file(args.class_names_dict_path)
    names_dict['dup'] = names_dict['bg']
    names_dict.pop('bg')
    names_dict['bg'] = names_dict['dup'] + 1
    names_dict = {key:value-1 for key, value in names_dict.items()}
    dup_cls = names_dict['dup']

    names_list = list(names_dict) 
    class_names = np.array(names_list)

    save_name = utils.get_value_in_pattern(args.src_path, '.*/(.*).json')

    cm_gt_path = os.path.join(args.save_dir, 'cm', save_name + '_cm_gt.npy')
    cm_pred_path = os.path.join(args.save_dir, 'cm', save_name + '_cm_pred.npy')
    if args.target_class : 
        png_path = os.path.join(args.save_dir, 'cm', save_name + '_' + args.target_class + '.png')
    else :
        png_path = os.path.join(args.save_dir, 'cm', save_name + '.png')

    if args.target_class :
        target_cls = [names_dict[args.target_class]] 
    #target_cls = [names_dict['coca2']]

    if args.load :
        cm_gt = np.load(cm_gt_path)
        cm_pred = np.load(cm_pred_path)

    else :
        bg_cls = names_dict['bg']
        cm_pred, cm_gt = [], []

        src_json = json_maker([], args.src_path, 0)
        gt_json = json_maker([], args.gt_path, 0)

        src_json.load()
        gt_json.load()

        scene_ids = src_json.get_all_scenes()
        cam_ids = [str(cam_id+1) for cam_id in range(args.num_cam)]

        T = {cls : [] for cls in class_names}
        P = {cls : [] for cls in class_names}

        for i, scene_id in tqdm(enumerate(scene_ids)) :
            #if not scene_id == '20190923-00007-03' : continue
            #if not i == 1 : continue
            gt_instance_summary = gt_json.get_instance_summary(scene_id)
            scene = src_json.get_scene(scene_id)
            for cam_id in cam_ids :
                #if not cam_id == '3' : continue
                src_cam = src_json.get_cam(scene_id, cam_id)
                pred_path, pred_insts = src_json.get_all_inst(src_cam, args.thresh) 
                gt_cam = gt_json.get_cam(scene_id, cam_id)
                gt_path, gt_insts = gt_json.get_all_inst(gt_cam)

                if(pred_path != gt_path):
                    exit(1)

                gt_ids = list(gt_json.get_all_insts_in_cam(scene_id, cam_id).keys()) 
                pred_ids = list(src_json.get_all_insts_in_cam(scene_id, cam_id).keys()) 
                pred_edge_dict = {pred_id : [] for pred_id  in pred_ids}
                gt_edge_dict = {gt_id : [] for gt_id  in gt_ids}
                pred_wgt_dict = {pred_id : [] for pred_id in pred_ids}
                gt_wgt_dict = {gt_id : [] for gt_id in gt_ids}
                for pred_inst in pred_insts :
                    for gt_inst in gt_insts :
                        wgt = utility.iou(pred_inst['pos'], gt_inst['pos'])
                        pred_id = pred_inst['inst_id']
                        gt_id = gt_inst['inst_id']
                        if wgt > .5: 
                            pred_edge_dict[pred_id].append(gt_id) 
                            gt_edge_dict[gt_id].append(pred_id) 
                            pred_wgt_dict[pred_id].append(wgt) 
                            gt_wgt_dict[gt_id].append(wgt) 
            
                is_gt_seen = {gt_id : 0 for gt_id in gt_ids}
                for pred_inst in pred_insts :
                    pred_id = pred_inst['inst_id']
                    pred_cls = pred_inst['subcls']
                    pred_cls_name = class_names[pred_cls]
                    cm_pred.append(pred_cls)
                    P[pred_cls_name].append(pred_inst['prob'])

                    matched_gt_ids = np.array(pred_edge_dict[pred_id])
                    matched_wgts = np.array(pred_wgt_dict[pred_id])
                    if matched_gt_ids.size :
                        gt_id = matched_gt_ids[np.argmax(matched_wgts)]
                        gt_subcls = gt_instance_summary[gt_id] -1
                        gt_cls_name = class_names[gt_subcls]
                        if gt_subcls == pred_cls :
                            if is_gt_seen[gt_id] : 
                                gt_subcls = dup_cls
                                T[pred_cls_name].append(0)
                            else :
                                is_gt_seen[gt_id] = 1 
                                T[pred_cls_name].append(1)
                        else :
                            T[pred_cls_name].append(0)
                    else :
                        gt_subcls = bg_cls
                        T[pred_cls_name].append(0)

                    cm_gt.append(gt_subcls)

                for gt_id, is_seen in is_gt_seen.items() :
                    if not is_seen :
                        gt_subcls = gt_instance_summary[gt_id] -1
                        gt_cls_name = class_names[gt_subcls]
                        cm_gt.append(gt_subcls)
                        cm_pred.append(bg_cls)

                        T[gt_cls_name].append(1)
                        P[gt_cls_name].append(0)
                #break
            #if i == 1 : break
            #break
            #if i == 50 : break

        cm_gt, cm_pred = zip(*sorted(zip(cm_gt, cm_pred), key = lambda ele : ele[0]))
        cm_gt, cm_pred = map(np.array, [cm_gt, cm_pred])


        np.save(cm_gt_path, cm_gt)
        np.save(cm_pred_path, cm_pred)

    if args.target_class : 
        #target_idx = np.where(cm_gt == target_cls )
        target_idx = np.union1d(np.where(cm_gt == target_cls), np.where(cm_pred == target_cls ))
        cm_gt = cm_gt[target_idx]
        cm_pred = cm_pred[target_idx]
       
    cm = confusion_matrix(cm_gt, cm_pred)

    if args.target_class : 
        #gt_classes = class_names[target_cls]
        gt_classes = class_names[unique_labels(cm_gt, cm_pred)]
        pred_classes =  class_names[unique_labels(cm_gt, cm_pred)]
        
        cm = cm[np.where(pred_classes == gt_classes)]

    else :
        gt_classes = class_names[unique_labels(cm_gt, cm_pred)]
        pred_classes =  class_names[unique_labels(cm_gt, cm_pred)]

    #plot_confusion_matrix(cm, cm_gt, cm_pred, gt_classes, pred_classes, png_path, args.text, normalize=False)


    #pr_curve_save_dir = os.path.join(args.save_dir, 'pr_curve', save_name)
    #os.makedirs(pr_curve_save_dir, exist_ok = True)
    #draw_precision_recall_curve(T, P, gt_classes, pr_curve_save_dir)
    calc_map(T, P, gt_classes)
    #print_TF(cm, gt_classes)




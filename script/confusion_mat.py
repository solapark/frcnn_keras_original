from tqdm import tqdm

import utility, utils
from bipartite_graph import Bipartite_graph
from json_maker import json_maker

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

import argparse 
import sys

#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, default='/data1/sap/MessyTable/labels/test.json')
    parser.add_argument('--src_path', type=str, default='/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log.json')
    #parser.add_argument('--dst_path', type=str, default='/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet.json')
    parser.add_argument('--class_names_dict_path', type=str, default='/data1/sap/MessyTable/class.dict')

    args = parser.parse_args()

    names_dict = utils.get_dict_from_file(args.class_names_dict_path)
    names_dict = {key:value-1 for key, value in names_dict.items()}
    cm_pred, cm_gt = [], []
    bg_cls = names_dict['bg']

    src_json = json_maker([], args.src_path, 0)
    gt_json = json_maker([], args.gt_path, 0)
    #dst_json = json_maker([], args.dst_path, 0)

    src_json.load()
    gt_json.load()

    #dst_json.insert_intrinsics(gt_json.get_intrinsics())
    scene_ids = src_json.get_all_scenes()
    for scene_id in tqdm(scene_ids) :
        #G = Bipartite_graph()
        instance_summary = gt_json.get_instance_summary(scene_id) 
        #instance_ids = [int(id) for id in instance_summary.keys()]
        #new_id = max(instance_ids)+1

        #dst_json.insert_scene(scene_id)
        scene = src_json.get_scene(scene_id)
        cam_ids = src_json.get_all_cams(scene)
        for cam_id in cam_ids :
            G = Bipartite_graph()
            #dst_json.insert_cam(scene_id, cam_id)

            src_cam = src_json.get_cam(scene_id, cam_id)
            pred_path, pred_insts = src_json.get_all_inst(src_cam) 

            gt_cam = gt_json.get_cam(scene_id, cam_id)
            gt_path, gt_insts = gt_json.get_all_inst(gt_cam)

            if(pred_path != gt_path):
                #print('pred_path', pred_path, 'gt_path', gt_path)
                exit(1)

            #dst_json.insert_path(scene_id, cam_id, pred_path)

            #extrinsics = gt_json.get_extrinsics(scene_id, cam_id)
            #dst_json.insert_extrinsics(scene_id, cam_id, extrinsics)

            edges = []
            for pred_inst in pred_insts :
                for gt_inst in gt_insts :
                    #pred_inst_pos = [p/resize_ratio for p in pred_inst['pos']]
                    #wgt = utility.iou(pred_inst_pos, gt_inst['pos'])
                    wgt = utility.iou(pred_inst['pos'], gt_inst['pos'])
                    pred_id_in_G = 'pred_' + pred_inst['inst_id']
                    gt_id_in_G = 'gt_' + gt_inst['inst_id']
                    if wgt : edges.append((pred_id_in_G, gt_id_in_G, wgt))
        
            G.add_weighted_edges(edges)
            matching = G.match()
            matching = [sorted([k, v], reverse=True) for (k, v) in matching.items()]
            matching = {l[0]:l[1] for l in matching}

            gt_ids = list(instance_summary.keys())
            for pred_inst in pred_insts :
                pred_id = pred_inst['inst_id']
                pred_id_in_G = 'pred_' + pred_id
                pred_cls = pred_inst['subcls']
                cm_pred.append(pred_cls)
                if pred_id_in_G in matching :
                    gt_id_in_G = matching[pred_id_in_G]
                    dst_inst_id = gt_id_in_G.split('_')[1]
                    gt_subcls = instance_summary[dst_inst_id] -1
                    gt_ids.remove(dst_inst_id)
                else :
                    gt_subcls = bg_cls

            for gt_id in gt_ids :
                cm_gt.append(gt_subcls)
                cm_pred.append(bg_cls)

    names_list = list(names_dict) 
    class_names = np.array(names_list)
    print(confusion_matrix(cm_truth, cm_pred))
    plot_confusion_matrix(cm_truth, cm_pred, class_names)

    plt.rcParams["figure.figsize"] = (500, 500)
    #plt.figure(figsize= (50, 50))
    plt.savefig('test.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    return ax



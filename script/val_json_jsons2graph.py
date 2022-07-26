import utils
import os
import numpy as np
import copy

folder = '/data3/sap/frcnn_keras_original/experiment/val_json_json'
#file_name = 'sv_messytable_*/log_val_json_json.txt'
#file_name_pattern = 'sv_messytable_(.*)/log_val_json_json.txt'

#file_name = 'sv_messytable_from_1041_*/log_val_json_json.txt'
#file_name_pattern = 'sv_messytable_from_1041_(.*)/log_val_json_json.txt'

file_name = '220516/mv_messytable_fine_tunning_from_model9/val/MODA/model_*/log_val_json_json.txt'
file_name_pattern = '220516/mv_messytable_fine_tunning_from_model9/val/MODA/model_(.*)/log_val_json_json.txt'

metric_patterns = {
    'MODA' : 'ALL\t(.*)\t.*\t.*\t.*\t.*', 
    'MODP' : 'ALL\t.*\t(.*)\t.*\t.*\t.*',
    'F1' : 'ALL\t.*\t.*\t(.*)\t.*\t.*',
    'Recall' : 'ALL\t.*\t.*\t.*\t(.*)\t.*',
    'Precision' : 'ALL\t.*\t.*\t.*\t.*\t(.*)'
}
metric_line_num = 1

scatter_drawer = utils.Draw_scatter(len(metric_patterns), 'result.png')

full_path = os.path.join(folder, file_name)
full_pattern = os.path.join(folder, file_name_pattern)
file_list = utils.get_file_list_from_fileform(full_path)
file_list.sort()

epochs = []
metric_values = {metric_name : [] for metric_name in metric_patterns.keys()}
for f in file_list :
    epoch = utils.get_value_in_pattern(f, full_pattern)
    lines = utils.get_list_from_file(f)
    if len(lines) ==0 : continue
    
    is_valid_epoch = 1
    metric_line = utils.get_list_from_file(f)[metric_line_num]
    for metric_name, metric_pattern in metric_patterns.items():
        metric_value = utils.get_value_in_pattern(metric_line, metric_pattern)
        if not metric_value : 
            is_valid_epoch = 0
            break 
        metric_value = float(metric_value)
        if metric_value == -1 : continue
        metric_values[metric_name].append(float(metric_value))
        #print('%s\t%s\t%s'%(metric_name, epoch, metric_value)) 

    if is_valid_epoch : 
        epochs.append(int(epoch))

print('max')
for i, (metric_name, metric_value) in enumerate(metric_values.items()):
    cur_epochs = copy.deepcopy(epochs)
    cur_epochs, metric_value = zip(*sorted(zip(cur_epochs, metric_value)))
    max_value = max(metric_value)
    max_idx = metric_value.index(max_value)
    max_epoch = cur_epochs[max_idx]
    print(metric_name, ':', max_value, '@',  max_epoch)

    scatter_drawer.add(cur_epochs, metric_value, cmap_idx = i, label = metric_name)
    #for e, m in zip(cur_epochs, metric_value):
    #    print(e, ':', m)

scatter_drawer.show()
scatter_drawer.save()

'''
map_pattern = 'mAP\t(.*)'
full_path = os.path.join(folder, file_name)
full_pattern = os.path.join(folder, file_name_pattern)
file_list = utils.get_file_list_from_fileform(full_path)
file_list.sort()

epochs = []
mAPs = []
for f in file_list :
    epoch = utils.get_value_in_pattern(f, full_pattern)
    lines = utils.get_list_from_file(f)
    if len(lines) ==0 : continue
    map_line = utils.get_list_from_file(f)[2]
    mAP = utils.get_value_in_pattern(map_line, map_pattern)

    epochs.append(int(epoch))
    mAPs.append(float(mAP))

    print('%s\t%s'%(epoch, mAP)) 

max_mAP = max(mAPs)
max_idx = mAPs.index(max_mAP)
max_epoch = epochs[max_idx]
print('max')
print(max_epoch, '\t', max_mAP)

utils.scatter(epochs, mAPs)
'''

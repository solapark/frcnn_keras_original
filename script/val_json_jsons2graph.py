import utils
import os

folder = '/data3/sap/frcnn_keras_original/experiment/val_json_json'
file_name = 'sv_messytable_*/log_val_json_json.txt'
file_name_pattern = 'sv_messytable_(.*)/log_val_json_json.txt'

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

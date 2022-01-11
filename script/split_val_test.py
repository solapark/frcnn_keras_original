from utils import csv2list, split_list, list2csv

all_path = '/data1/sap/frcnn_keras/data/mv_test_backup.txt'
l1_path = '/data1/sap/frcnn_keras/data/mv_val.txt'
l2_path = '/data1/sap/frcnn_keras/data/mv_test.txt'
ratio = .5

all_list = csv2list(all_path)
size = int(len(all_list)*ratio)
l1, l2 = split_list(all_list, size)

l1 = sorted(l1)
l2 = sorted(l2)

list2csv(l1_path, l1)
list2csv(l2_path, l2)

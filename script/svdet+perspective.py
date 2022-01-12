import networkx as nx
import json
from itertools import combinations 
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
from tqdm import tqdm

from bipartite_graph import Bipartite_graph
from json_maker import json_maker

src_json_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log_with_gt_inst_id.json'
dst_json_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+perspective.json'
num_cam = 3

import numpy as np
import cv2
 
def draw_circle(img, org, radius, color, thickness): 
    img = cv2.circle(img, tuple(org), radius, color, thickness)

imageA_path = '/data1/sap/MessyTable/images/20190921-00003-01-01.jpg' 
imageB_path = '/data1/sap/MessyTable/images/20190921-00003-01-02.jpg' 
imageA = cv2.imread(imageA_path) 
imageB = cv2.imread(imageB_path)
 
src = np.float32(
                        [
                            [
                                (103+518)/2, 
                                707,
                            ],
                            [
                                (368+624)/2,
                                452,
                            ],
                            [
                                (532+788)/2,
                                657,
                            ],
                            [
                                (627+831)/2,
                                302,
                            ]
                        ]
).reshape((-1, 1, 2))

dst = np.float32(
                        [
                            [
                                (300+640)/2,
                                1047,
                            ],
                            [
                                (663+989)/2,
                                731,
                            ],
                            [
                                (793+1151)/2,
                                1080,
                            ],
                            [
                                (1093+1313)/2,
                                467,
                            ]
                        ]
).reshape((-1, 1, 2))

H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
 
before = []
point = [830, 286, 1]
before.append(point)
before = np.array(before).transpose() #(3, N)
 
after = np.matmul(H, before)
after = after / after[2, :]
after = after[:2, :] #(2, N)
after = np.round(after, 0).astype(np.int)
 
draw_circle(imageA, tuple(before[:2, 0]), 20, (255, 0, 0), 2)
draw_circle(imageB, tuple(after[:, 0]), 20, (255, 0, 0), 2)

#draw_circle(imageA, (819, 1623), 20, (255, 0, 0), 2)
#draw_circle(imageB, (670, 507), 20, (255, 0, 0), 2)
cv2.imshow('imageA', imageA) 
cv2.imshow('imageB', imageB) 
cv2.waitKey()



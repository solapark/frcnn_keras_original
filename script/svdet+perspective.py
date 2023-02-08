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
 
src = np.float32([
                        [
                            [
                                1623.0,
                                819.0
                            ],
                            [
                                1454.0,
                                802.0
                            ],
                            [
                                1468.0,
                                637.0
                            ],
                            [
                                1635.0,
                                653.0
                            ]
                        ],
                        [
                            [
                                1314.0,
                                787.0
                            ],
                            [
                                1149.0,
                                771.0
                            ],
                            [
                                1163.0,
                                608.0
                            ],
                            [
                                1327.0,
                                623.0
                            ]
                        ],
                        [
                            [
                                1011.0,
                                757.0
                            ],
                            [
                                850.0,
                                741.0
                            ],
                            [
                                865.0,
                                580.0
                            ],
                            [
                                1026.0,
                                596.0
                            ]
                        ],
                        [
                            [
                                1653.0,
                                415.0
                            ],
                            [
                                1487.0,
                                400.0
                            ],
                            [
                                1499.0,
                                240.0
                            ],
                            [
                                1664.0,
                                253.0
                            ]
                        ],
                        [
                            [
                                1346.0,
                                389.0
                            ],
                            [
                                1184.0,
                                374.0
                            ],
                            [
                                1198.0,
                                215.0
                            ],
                            [
                                1360.0,
                                229.0
                            ]
                        ],
                        [
                            [
                                1048.0,
                                363.0
                            ],
                            [
                                887.0,
                                348.0
                            ],
                            [
                                903.0,
                                190.0
                            ],
                            [
                                1062.0,
                                203.0
                            ]
                        ]
                    ],
).reshape((-1, 1, 2))

dst = np.float32([
                        [
                            [
                                507.0,
                                670.0
                            ],
                            [
                                287.0,
                                736.0
                            ],
                            [
                                206.0,
                                533.0
                            ],
                            [
                                426.0,
                                490.0
                            ]
                        ],
                        [
                            [
                                816.0,
                                577.0
                            ],
                            [
                                662.0,
                                623.0
                            ],
                            [
                                582.0,
                                460.0
                            ],
                            [
                                737.0,
                                428.0
                            ]
                        ],
                        [
                            [
                                1040.0,
                                510.0
                            ],
                            [
                                927.0,
                                543.0
                            ],
                            [
                                851.0,
                                406.0
                            ],
                            [
                                966.0,
                                383.0
                            ]
                        ],
                        [
                            [
                                864.0,
                                210.0
                            ],
                            [
                                746.0,
                                218.0
                            ],
                            [
                                680.0,
                                100.0
                            ],
                            [
                                797.0,
                                100.0
                            ]
                        ],
                        [
                            [
                                630.0,
                                228.0
                            ],
                            [
                                473.0,
                                239.0
                            ],
                            [
                                405.0,
                                102.0
                            ],
                            [
                                562.0,
                                101.0
                            ]
                        ],
                        [
                            [
                                318.0,
                                251.0
                            ],
                            [
                                99.0,
                                268.0
                            ],
                            [
                                31.0,
                                100.0
                            ],
                            [
                                249.0,
                                101.0
                            ]
                        ]
                    ],
).reshape((-1, 1, 2))

H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
 
before = []
#point = [830, 286, 1]
#point = [1623, 819, 1]
#point = [1653, 415, 1]
point = [922, 547, 1]

before.append(point)
before = np.array(before).transpose() #(3, N)
 
after = np.matmul(H, before)
after = after / after[2, :]
after = after[:2, :] #(2, N)
after = np.round(after, 0).astype(np.int)
 
print(before, after)
draw_circle(imageA, tuple(before[:2, 0]), 20, (255, 0, 0), 2)
draw_circle(imageB, tuple(after[:, 0]), 20, (255, 0, 0), 2)

#draw_circle(imageA, (819, 1623), 20, (255, 0, 0), 2)
#draw_circle(imageB, (670, 507), 20, (255, 0, 0), 2)
cv2.imwrite('imageA.png', imageA) 
cv2.imwrite('imageB.png', imageB) 
cv2.waitKey()



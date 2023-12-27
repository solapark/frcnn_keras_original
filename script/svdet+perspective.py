#src_json_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/log_with_gt_inst_id.json'
#dst_json_path = '/data3/sap/frcnn_keras/result/result-sv_messytable_cam3_resume_model_110/test_3cam/svdet+perspective.json'

import numpy as np
import cv2

def draw_circle(img, org, radius, color, thickness): 
    img = cv2.circle(img, tuple(org), radius, color, thickness)


imageA_path = '/data1/sap/MessyTable/images/20190921-00003-01-01.jpg' 
imageB_path = '/data1/sap/MessyTable/images/20190921-00003-01-03.jpg' 
#imageB_path = '/data1/sap/MessyTable/images/20190921-00003-01-02.jpg' 
#imageB_path = '/data1/sap/MessyTable/images/20190921-00003-01-05.jpg' 
imageA = cv2.imread(imageA_path) 
imageB = cv2.imread(imageB_path)
 
cam1 = np.float32([
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

cam5 = np.float32([
                        [
                            [
                                1511.0,
                                443.0
                            ],
                            [
                                1446.0,
                                287.0
                            ],
                            [
                                1640.0,
                                241.0
                            ],
                            [
                                1721.0,
                                397.0
                            ]
                        ],
                        [
                            [
                                1087.0,
                                920.0
                            ],
                            [
                                1049.0,
                                710.0
                            ],
                            [
                                1258.0,
                                665.0
                            ],
                            [
                                1316.0,
                                875.0
                            ]
                        ],
                        [
                            [
                                1662.0,
                                807.0
                            ],
                            [
                                1575.0,
                                597.0
                            ],
                            [
                                1801.0,
                                550.0
                            ],
                            [
                                1910.0,
                                760.0
                            ]
                        ],
                        [
                            [
                                1020.0,
                                556.0
                            ],
                            [
                                991.0,
                                398.0
                            ],
                            [
                                1173.0,
                                354.0
                            ],
                            [
                                1216.0,
                                511.0
                            ]
                        ],
                        [
                            [
                                969.0,
                                279.0
                            ],
                            [
                                946.0,
                                157.0
                            ],
                            [
                                1106.0,
                                113.0
                            ],
                            [
                                1140.0,
                                236.0
                            ]
                        ],
                        [
                            [
                                1398.0,
                                171.0
                            ],
                            [
                                1347.0,
                                49.0
                            ],
                            [
                                1517.0,
                                4.0
                            ],
                            [
                                1580.0,
                                125.0
                            ]
                        ]
                    ],
).reshape((-1, 1, 2))

cam2 = np.float32([
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

cam3 = np.float32([
                        [
                            [
                                1715.0,
                                904.0
                            ],
                            [
                                1441.0,
                                935.0
                            ],
                            [
                                1353.0,
                                701.0
                            ],
                            [
                                1596.0,
                                677.0
                            ]
                        ],
                        [
                            [
                                666.0,
                                1027.0
                            ],
                            [
                                359.0,
                                1063.0
                            ],
                            [
                                395.0,
                                798.0
                            ],
                            [
                                665.0,
                                772.0
                            ]
                        ],
                        [
                            [
                                1205.0,
                                963.0
                            ],
                            [
                                917.0,
                                998.0
                            ],
                            [
                                886.0,
                                749.0
                            ],
                            [
                                1142.0,
                                722.0
                            ]
                        ],
                        [
                            [
                                663.0,
                                493.0
                            ],
                            [
                                432.0,
                                511.0
                            ],
                            [
                                453.0,
                                358.0
                            ],
                            [
                                663.0,
                                344.0
                            ]
                        ],
                        [
                            [
                                1071.0,
                                459.0
                            ],
                            [
                                852.0,
                                476.0
                            ],
                            [
                                835.0,
                                331.0
                            ],
                            [
                                1035.0,
                                317.0
                            ]
                        ],
                        [
                            [
                                1464.0,
                                425.0
                            ],
                            [
                                1254.0,
                                442.0
                            ],
                            [
                                1202.0,
                                304.0
                            ],
                            [
                                1393.0,
                                291.0
                            ]
                        ],
                    ],
).reshape((-1, 1, 2))

src = cam1
#dst = cam2
dst = cam3

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

cv2.imwrite('imageA.png', imageA) 
cv2.imwrite('imageB.png', imageB) 



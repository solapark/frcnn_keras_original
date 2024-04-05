import cv2
import numpy as np
import copy

class AUGMENT:
    def __init__(self, args):
        self.args = args
    
    def augment(self, imgs, pred, gt, is_valid_batch):
        #img #(3, 1, W, H, 512)
        #pred (N, 3, 4) x1y1x2y2
        #gt (1, N) 
        rows, cols = imgs[0][0].shape[:2]
        gt_rows, gt_cols = rows*self.args.rpn_stride, cols*self.args.rpn_stride

        img_copy_list = [img[0].copy() for img in imgs]
        pred_copy = pred.copy()
        gt_copy = copy.deepcopy(gt[0])

        #self.draw(img_copy_list, pred_copy, gt_copy, '/home/sap/bf_aug.png')

        if self.args.hf and np.random.randint(0, 2) == 0:
            #print('hf')
            img_copy_list = [cv2.flip(img_copy, 1) for img_copy in img_copy_list]
            pred_copy[:, :, :, [0, 2]] = cols - pred_copy[:, :, :, [2, 0]]

            for inst in gt_copy:
                for cam_id, box in inst['resized_box'].items() :
                    x1, y1, x2, y2 = box
                    box[2] = gt_cols - x1
                    box[0] = gt_cols - x2
                    inst['resized_box'][cam_id] = box
            
        if self.args.vf and np.random.randint(0, 2) == 0:
            #print('vf')
            img_copy_list = [cv2.flip(img_copy, 0) for img_copy in img_copy_list]
            pred_copy[:, :, :, [1, 3]] = rows - pred_copy[:, :, :, [3, 1]]

            for inst in gt_copy:
                for cam_id, box in inst['resized_box'].items() :
                    x1, y1, x2, y2 = box
                    box[3] = gt_rows - y1
                    box[1] = gt_rows - y2
                    inst['resized_box'][cam_id] = box

        if self.args.rot:
            angle = np.random.choice([0,90,180,270],1)[0]
            if angle !=0 : 
                if angle == 270:
                    #print('270')
                    img_copy_list = [np.transpose(img_copy, (1,0,2)) for img_copy in img_copy_list]
                    img_copy_list = [cv2.flip(img_copy, 0) for img_copy in img_copy_list]
                elif angle == 180:
                    #print('180')
                    img_copy_list = [cv2.flip(img_copy, -1) for img_copy in img_copy_list]
                elif angle == 90:
                    #print('90')
                    img_copy_list = [np.transpose(img_copy, (1,0,2)) for img_copy in img_copy_list]
                    img_copy_list = [cv2.flip(img_copy, 1) for img_copy in img_copy_list]

                pred = pred_copy.copy()
                x1 = pred[..., 0]
                x2 = pred[..., 2]
                y1 = pred[..., 1]
                y2 = pred[..., 3]
                if angle == 270:
                    pred_copy[..., 0] = y1
                    pred_copy[..., 2] = y2
                    pred_copy[..., 1] = cols - x2
                    pred_copy[..., 3] = cols - x1
                elif angle == 180:
                    pred_copy[..., 2] = cols - x1
                    pred_copy[..., 0] = cols - x2
                    pred_copy[..., 3] = rows - y1
                    pred_copy[..., 1] = rows - y2
                elif angle == 90:
                    pred_copy[..., 0] = rows - y2
                    pred_copy[..., 2] = rows - y1
                    pred_copy[..., 1] = x1
                    pred_copy[..., 3] = x2        

                for inst in gt_copy:
                    for cam_id, box in inst['resized_box'].items() :
                        x1, y1, x2, y2 = box

                        if angle == 270:
                            box[0] = y1
                            box[2] = y2
                            box[1] = gt_cols - x2
                            box[3] = gt_cols - x1
                        elif angle == 180:
                            box[2] = gt_cols - x1
                            box[0] = gt_cols - x2
                            box[3] = gt_rows - y1
                            box[1] = gt_rows - y2
                        elif angle == 90:
                            box[0] = gt_rows - y2
                            box[2] = gt_rows - y1
                            box[1] = x1
                            box[3] = x2        

                        inst['resized_box'][cam_id] = box

        pred_copy[np.where(1-is_valid_batch)]=-1

        #self.draw(img_copy_list, pred_copy, gt_copy, '/home/sap/af_aug.png')

        img_copy_list = [np.expand_dims(img_copy, 0) for img_copy in img_copy_list]
        #print('')
        return img_copy_list, pred_copy, [gt_copy]

    def draw(self, img, pred, gt_copy, filename) :
        cam_idx=0

        filepath = '/data1/sap/MessyTable/images/20190921-00003-01-%02d.jpg'%(cam_idx+1)
        img_size = (1056, 592)
        img = cv2.imread(filepath)
        resized_image = cv2.resize(img, img_size)

        #resized_image = (img[cam_idx][..., :3]*255).astype(np.uint8)
        #rows, cols = resized_image.shape[:2]
        #resized_image = cv2.resize(resized_image, (cols*16, rows*16))

        colors = [
            (255, 0, 0),     # Red
            (255, 165, 0),   # Orange
            (255, 255, 0),   # Yellow
            (0, 255, 0),     # Lime
            (0, 128, 0),     # Green
            (0, 255, 255),   # Cyan
            (0, 0, 255),     # Blue
            (128, 0, 128),   # Purple
            (255, 0, 255),   # Magenta
            (255, 192, 203)  # Pink
        ]

        cnt = 0
        '''
        for inst in gt:
            if cnt==5 : break
            if 0 not in inst['resized_box'] : continue
            x1, y1, x2, y2 = map(int, inst['resized_box'][cam_idx])
            cv2.rectangle(resized_image, (x1, y1), (x2, y2), colors[cnt], 2)
            cnt+=1

        '''
        for box in pred[0] :
            if cnt==10 : break
            x1, y1, x2, y2 = map(int, box[cam_idx])
            if x1 == -1 : continue
            cv2.rectangle(resized_image, (x1*16, y1*16), (x2*16, y2*16), colors[cnt], 2)
            cnt+=1

        cv2.imwrite(filename, resized_image)

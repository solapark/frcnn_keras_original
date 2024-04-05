import numpy as np

class CALC_REGR:
    def __init__(self, std):
        self.std = np.array(std).reshape(-1, 1)

    def calc_t(self, pred, gt, H_filp=[], W_flip=[], Rot=[]):
        '''
        (cx_pred, cy_pred) = (pred[:, 1]+pred[:, 3])/2.0

        for i, (h_flip, w_flip, rot) in enumerate(H_flip, W_flip, Rot) : 
            if h_flip : 
                pred[i] = self.H_flip(pred[i], (cx_pred[i], cy_pred[i]))
                gt[i] = self.H_flip(gt[i], (cx_pred[i], cy_pred[i]))

            if w_flip : 
                pred[i] = self.W_flip(pred[i], (cx_pred[i], cy_pred[i]))
                gt[i] = self.W_flip(gt[i], (cx_pred[i], cy_pred[i]))

            if rot==90 : 
                pred[i] = self.Rot90(pred[i], (cx_pred[i], cy_pred[i]))
                gt[i] = self.Rot90(gt[i], (cx_pred[i], cy_pred[i]))

            if rot==180 : 
                pred[i] = self.Rot180(pred[i], (cx_pred[i], cy_pred[i]))
                gt[i] = self.Rot180(gt[i], (cx_pred[i], cy_pred[i]))

            if rot==270 : 
                pred[i] = self.Rot270(pred[i], (cx_pred[i], cy_pred[i]))
                gt[i] = self.Rot270(gt[i], (cx_pred[i], cy_pred[i]))
        '''

        (cx_gt, cy_gt), (cx_pred, cy_pred) = map(lambda a : [(a[:, 0]+a[:, 2])/2.0, (a[:, 1]+a[:, 3])/2.0], [gt, pred])
        (w_gt, h_gt), (w_pred, h_pred) = map(lambda a : [a[:, 2]-a[:, 0], a[:, 3]-a[:, 1]], [gt, pred])

        tx = (cx_gt - cx_pred) / w_pred
        ty = (cy_gt - cy_pred) / h_pred

        tw = np.log(w_gt/w_pred)
        th = np.log(h_gt/h_pred)

        tx[np.where(w_pred ==0)] = 0
        ty[np.where(h_pred ==0)] = 0
        tw[np.where(w_gt ==0)] = 0
        th[np.where(h_gt ==0)] = 0
    
        tx, ty, tw, th = self.std * [tx, ty, tw, th]

        return np.column_stack([tx, ty, tw, th])

    def H_flip(self, points, center):
        flipped_points = np.copy(points)
        flipped_points[:, 0] = 2 * center[0] - points[:, 0]
        flipped_points[:, 2] = 2 * center[0] - points[:, 2]
        return flipped_points

    def W_flip(points, center):
        flipped_points = np.copy(points)
        flipped_points[:, 1] = 2 * center[1] - points[:, 1]
        flipped_points[:, 3] = 2 * center[1] - points[:, 3]
        return flipped_points

    def Rot90(self, points, center):
        rotated_points = np.copy(points)
        rotated_points[:, 0] = center[1] + center[1] - points[:, 1]
        rotated_points[:, 1] = points[:, 0]
        rotated_points[:, 2] = center[1] + center[1] - points[:, 3]
        rotated_points[:, 3] = points[:, 2]
        return rotated_points

    def Rot180(self, points, center):
        rotated_points = np.copy(points)
        rotated_points[:, 0] = center[0] + center[0] - points[:, 0]
        rotated_points[:, 1] = center[1] + center[1] - points[:, 1]
        rotated_points[:, 2] = center[0] + center[0] - points[:, 2]
        rotated_points[:, 3] = center[1] + center[1] - points[:, 3]
        return rotated_points

    def Rot270(self, points, center):
        rotated_points = np.copy(points)
        rotated_points[:, 0] = points[:, 3]
        rotated_points[:, 1] = center[0] + center[0] - points[:, 2]
        rotated_points[:, 2] = points[:, 1]
        rotated_points[:, 3] = center[0] + center[0] - points[:, 0]
        return rotated_points


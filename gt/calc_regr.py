import numpy as np

class CALC_REGR:
    def __init__(self, std):
        self.std = np.array(std).reshape(-1, 1)

    def calc_t(self, pred, gt):
        (cx_gt, cy_gt), (cx_pred, cy_pred) = map(lambda a : [(a[:, 0]+a[:, 2])/2.0, (a[:, 1]+a[:, 3])/2.0], [gt, pred])
        (w_gt, h_gt), (w_pred, h_pred) = map(lambda a : [a[:, 2]-a[:, 0], a[:, 3]-a[:, 1]], [gt, pred])

        tx = (cx_gt - cx_pred) / w_pred
        ty = (cy_gt - cy_pred) / h_pred
        if((w_gt==0).any()):
            print('warning')

        tw = np.log(w_gt/w_pred)
        th = np.log(h_gt/h_pred)

        tx, ty, tw, th = self.std * [tx, ty, tw, th]

        return np.column_stack([tx, ty, tw, th])



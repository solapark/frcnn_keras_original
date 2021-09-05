from keras import backend as K
from keras.objectives import categorical_crossentropy

if K.common.image_dim_ordering() == 'tf':
	import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

def loss_dummy(y_true, y_pred):
    return tf.constant([0], dtype=tf.float32)

def rpn_loss_regr(num_anchors):
	def rpn_loss_regr_fixed_num(y_true, y_pred):
		if K.common.image_dim_ordering() == 'th':
			x = y_true[:, 4 * num_anchors:, :, :] - y_pred
			x_abs = K.abs(x)
			x_bool = K.less_equal(x_abs, 1.0)
			return lambda_rpn_regr * K.sum(
				y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :4 * num_anchors, :, :])
		else:
			x = y_true[:, :, :, 4 * num_anchors:] - y_pred
			x_abs = K.abs(x)
			x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

			return lambda_rpn_regr * K.sum(
				y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

	return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
	def rpn_loss_cls_fixed_num(y_true, y_pred):
		if K.common.image_dim_ordering() == 'tf':
			return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
		else:
			return lambda_rpn_class * K.sum(y_true[:, :num_anchors, :, :] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, num_anchors:, :, :])) / K.sum(epsilon + y_true[:, :num_anchors, :, :])

	return rpn_loss_cls_fixed_num

def ven_loss(alpha=.3):
    """Loss function for rpn classification
    Args:
        y_pred: view invariant features, shape == (1, num_cam, 300, vi_featue_size)
        y_true: anchor_target idx, shape == (1, 3, num_GT, 4, 1, 1)
            first : batch_size(dummy)
            second : num_cam 
            third : num insta
            fourth : cam_idx, h idx, w idx, a idx
    Returns:
    """
    def triplet_loss_func(y_true, y_pred):
        y_true = y_true[0, :, :, :, 0, 0]
        y_true = tf.cast(y_true, 'int32') #(3, numSample, 4)
        anchor_idx = y_true[0] #(numSample, 4)
        pos_idx = y_true[1] #(numSample, 4)
        neg_idx = y_true[2] #(numSample, 4)

        anchor = tf.gather_nd(y_pred[0], anchor_idx) #(numSample, vi_feature_size)
        positive = tf.gather_nd(y_pred[0], pos_idx) #(numSample, vi_feature_size)
        negative = tf.gather_nd(y_pred[0], neg_idx) #(numSample, vi_feature_size)

        positive_dist = tf.sqrt(tf.reduce_sum(tf.square(anchor - positive), -1)) #(numSample, )
        negative_dist = tf.sqrt(tf.reduce_sum(tf.square(anchor - negative), -1)) #(numSample, )

        loss_1 = positive_dist - negative_dist + alpha
        #loss = tf.reduce_sum(tf.maximum(loss_1, 0.0)) #()
        loss = tf.reduce_mean(tf.maximum(loss_1, 0.0)) #()

        return loss
    return triplet_loss_func

def class_loss_regr(num_classes, num_cam):
    """Loss function for rpn regression
    Args:
        num_anchors: number of anchors (9 in here)
        num_cam : number of cam (3 in here)
    Returns:
        Smooth L1 loss function 
                           0.5*x*x (if x_abs < 1)
                           x_abx - 0.5 (otherwise)
    """
    def class_loss_regr_fixed_num(y_true, y_pred):
        #x = y_true[:, :, 4*num_classes:] - y_pred
        x = y_true[:, :, num_cam*4*num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        #return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
        return lambda_cls_regr * K.sum(y_true[:, :, :num_cam*4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :num_cam*4*num_classes])
        #return lambda_cls_regr * K.sum(y_true[:, :, :num_cam*4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :num_cam*4*num_classes]) * 0
    return class_loss_regr_fixed_num

def class_loss_cls(y_true, y_pred):
    return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))

#from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, \
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout, Reshape, Lambda, Concatenate
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed

from keras import backend as K

from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras_frcnn.FixedBatchNormalization import FixedBatchNormalization

def get_weight_path():
    return 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'

def make_model(args):
    return FRCNN_VGG(args)

class FRCNN_VGG:
    def __init__(self, args):
        self.args = args

    def nn_base_model(self):
        model = Sequential(name='nn_base')

        # Block 1
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        # Block 2
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

        # Block 3
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        # Block 4
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

        # Block 5
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
        # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        return model

    def rpn_layer_model(self, num_anchors):
        rpn_share = Sequential(name='rpn_body')
        rpn_share.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1'))
        model_class = Sequential(name='rpn_cls')
        model_regr = Sequential(name='rpn_regr')

        model_class.add(Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class'))
        model_regr.add(Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress'))

        return rpn_share, model_class, model_regr

    def view_invariant_layer_model(self, H, W, num_anchors, view_invar_feature_size):
        """Create a view invariant layer
            Step1: Pass through the feature map from rpn body layer to convolutional layers
                    Keep the padding 'same' to preserve the feature map's size
            Step2: Pass the step1 to two (1,1) convolutional layer to replace the fully connected layer. num_anchors*view_invar_feature_size (9*128 in here) channels for 0~1 sigmoid view invariant feature
        Args:
            num_anchors: 9 in here
            view_invar_feature_size

        Returns:
            view invariant feature 
        """
        model = Sequential(name='vi_layer')
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='vi_conv1'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='vi_conv2'))
        model.add(Conv2D(num_anchors * view_invar_feature_size, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='vi_out'))
        model.add(Reshape((H, W, num_anchors, view_invar_feature_size), name='vi_reshape'))
        model.add(Lambda(lambda x: K.l2_normalize(x, -1), name='vi_l2_norm'))
        model.add(Lambda(lambda x: K.expand_dims(x, 1), name='vi_expand_dim'))
        return model

    def view_invariant_conc_layer(self, view_invariants):
        #view_invariants x : list of (None, 1, H, W, C)
        return Concatenate(axis=1, name='vi_pooling')(view_invariants)
 

    def classifier_layer(self, base_layers, input_rois, num_rois, num_feat, num_cam, nb_classes):
        """Create a classifier layer
        
        Args:
            base_layers: list(vgg)
            input_rois: list(`(1,num_rois,4)` list of rois, with ordering (x,y,w,h))
            num_rois: number of rois to be processed in one time (4 in here)

        Returns:
            list(out_class, out_regr)
            out_class: classifier layer output
            out_regr: regression layer output
        """

        input_shape = (num_rois,7,7,512)
        pooling_regions = 7

        # num_rois (4) 7x7 roi pooling
        reduce_channel = Conv2D(num_feat//num_cam, (3, 3), activation='relu', padding='same', name='reduce_channel')
        out_roi_pools = []
        for i in range(num_cam):
            reduced_base_layer = reduce_channel(base_layers[i])
            out_roi_pools.append(RoiPoolingConv(pooling_regions, num_rois)([reduced_base_layer, input_rois[i]])) #(1, 4, 7, 7, 512)
            #out_roi_pools.append(RoiPoolingConv(pooling_regions, num_rois)([base_layers[i], input_rois[i]])) #(1, 4, 7, 7, 512)
        out_roi_pool = Concatenate(axis=-1, name='ViewPooling')(out_roi_pools)
        out_roi_pool_flat = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
        # Flatten the convlutional layer and connected to 2 FC and 2 dropout
        #out = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='classifier_conv1'))(out_roi_pool)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out_roi_pool_flat)
        out = TimeDistributed(Dropout(0.5))(out)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
        out = TimeDistributed(Dropout(0.5))(out)

        # There are two output layer
        # out_class: softmax acivation function for classify the class name of the object
        # out_regr: linear activation function for bboxes coordinates regression
        # note: no regression target for bg class
        out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
        out_regr = TimeDistributed(Dense(num_cam* 4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

        return [out_class, out_regr]

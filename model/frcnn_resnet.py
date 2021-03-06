from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, Convolution1D, MaxPooling2D, ZeroPadding2D, \
    AveragePooling2D, TimeDistributed, Lambda, Concatenate

from keras import backend as K

from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras_frcnn.FixedBatchNormalization import FixedBatchNormalization
import tensorflow as tf

def get_weight_path():
    return 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'

def make_model(args):
    return FRCNN_RESNET(args)

'''ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
'''

class FRCNN_RESNET:
    def __init__(self, args):
        self.args = args

    def nn_base(self, input_tensor=None, trainable=False):
        img_input = input_tensor
        bn_axis = 3
        x = ZeroPadding2D((3, 3))(img_input)

        x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable = trainable)(x)
        x = FixedBatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable = trainable)
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable = trainable)
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable = trainable)

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a', trainable = trainable)
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b', trainable = trainable)
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c', trainable = trainable)
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d', trainable = trainable)

        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a', trainable = trainable)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable = trainable)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable = trainable)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d', trainable = trainable)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e', trainable = trainable)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f', trainable = trainable)

        return x

    def classifier_layers(self, x, input_shape, trainable=False):

        x = self.conv_block_td(x, 3, [512, 512, 2048], stage=10, block='a', input_shape=input_shape, strides=(2, 2), trainable=trainable)

        x = self.identity_block_td(x, 3, [512, 512, 2048], stage=10, block='b', trainable=trainable)
        x = self.identity_block_td(x, 3, [512, 512, 2048], stage=10, block='c', trainable=trainable)
        x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)

        return x

    def ven(self, shared_layer, indices, ven_feat_size):
        '''
        input : shared_layer #(batch_size, H, W, A, C)
                indices #(batch_size, 300, 2)
                    contents of last axis 
                        first : H idx
                        Second : W idx
        output : view_embedding #(batch_size, 1, 300, ven_size)
        '''
        x = Lambda(lambda x: tf.gather_nd(x[0], x[1], batch_dims=1), name='ven_gather_nd')([shared_layer, indices])
        x = Convolution1D(512, 1, padding='same', activation='relu', kernel_initializer='normal', name='ven_conv1')(x)
        x = Convolution1D(512, 1, padding='same', activation='relu', kernel_initializer='normal', name='ven_conv2')(x)
        x = Convolution1D(ven_feat_size, 1, activation='sigmoid', kernel_initializer='uniform', name='ven_out')(x)
        x = Lambda(lambda x: K.l2_normalize(x, -1), name='ven_l2norm')(x)

        x = Lambda(lambda x: K.expand_dims(x, 1), name='vi_expand_dim')(x)
        return x

    def ven_conc(self, ven_outs):
        #view_invariants x : list of (B, 1, 300, C)
        return Concatenate(axis=1, name='ven_conc')(ven_outs)
 
 
    def rpn(self, base_layers,num_anchors):
        x = Convolution2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

        x_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
        x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

        return [x_class, x_regr, base_layers]

    def classifier(self, base_layers, input_rois, num_rois, num_cam, nb_classes, trainable=False):

        pooling_regions = 14
        input_shape = (num_rois,14,14,1024)

        reduce_channel = Convolution2D(512//num_cam, (3, 3), activation='relu', padding='same', name='reduce_channel')
        out_roi_pools = []
        for i in range(num_cam):
            reduced_base_layer = reduce_channel(base_layers[i])
            out_roi_pools.append(RoiPoolingConv(pooling_regions, num_rois)([reduced_base_layer, input_rois[i]])) #(1, 4, 7, 7, 512)
        out_roi_pool = Concatenate(axis=-1, name='ViewPooling')(out_roi_pools)
        out = self.classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
        out = TimeDistributed(Flatten())(out)
        out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='classifier_class_{}'.format(nb_classes))(out)
        out_regr = TimeDistributed(Dense(num_cam* 4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='classifier_regress_{}'.format(nb_classes))(out)
        return [out_class, out_regr]

    def get_img_output_length(self, width, height):
        def get_output_length(input_length):
            # zero_pad
            input_length += 6
            # apply 4 strided convolutions
            filter_sizes = [7, 3, 1, 1]
            stride = 2
            for filter_size in filter_sizes:
                input_length = (input_length - filter_size + stride) // stride
            return input_length

        return get_output_length(width), get_output_length(height) 

    def identity_block(self, input_tensor, kernel_size, filters, stage, block, trainable=True):

        nb_filter1, nb_filter2, nb_filter3 = filters
        
        if K.common.image_dim_ordering() == 'tf':
            bn_axis = 3
        else:
            bn_axis = 1

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Convolution2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=trainable)(input_tensor)
        x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
        x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
        x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        return x


    def identity_block_td(self, input_tensor, kernel_size, filters, stage, block, trainable=True):

        # identity block time distributed

        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.common.image_dim_ordering() == 'tf':
            bn_axis = 3
        else:
            bn_axis = 1

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
        x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',padding='same'), name=conv_name_base + '2b')(x)
        x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2c')(x)
        x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

        x = Add()([x, input_tensor])
        x = Activation('relu')(x)

        return x

    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):

        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.common.image_dim_ordering() == 'tf':
            bn_axis = 3
        else:
            bn_axis = 1

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Convolution2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', trainable=trainable)(input_tensor)
        x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
        x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
        x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = Convolution2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', trainable=trainable)(input_tensor)
        shortcut = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    def conv_block_td(self, input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):

        # conv block time distributed

        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.common.image_dim_ordering() == 'tf':
            bn_axis = 3
        else:
            bn_axis = 1

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
        x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2b')(x)
        x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c', trainable=trainable)(x)
        x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

        shortcut = TimeDistributed(Convolution2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
        shortcut = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

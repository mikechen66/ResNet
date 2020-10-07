#!/usr/bin/env python
# coding: utf-8

# resnet101.py

"""
ResNet101 model for Keras.

Please pay more attention on the formal argument "x". To faciliate the process of parameter passing
during the function calls in the context, we select x to express the recursion that is the typical
mathematical usage. 

Remember that it is the TensorFlow realization with image_data_foramt = 'channels_last'. If the env 
of Keras is 'channels_first', please change it according to the TensorFlow convention. Users can run
the client script to call the ResNet model.

$ python rn101_predict.py

Even adopting the validation_utils of imageNet and changing the prediction method in predict_val.py, 
the correctedness is extremely lower than the inception v4 model becuase the residual layer greatly 
increases the "raw" data. So it is subject to the brute force computing, i.e., update the moving
average from 100 to 1000 epochs before converging to the "real" mean and variance. That's why the 
prediction was wrong in the early stages with ResNet. Users can verify it by forcing the BatchNorm 
Layer to run in the "Training mode".

Custom Layer for ResNet used for BatchNormalization. Learns a set of weights and biases used for 
scaling the input data. The output consists simply in an element-wise multiplication of the input
and a sum of a set of constants:

    out = in*gamma + beta,

where 'gamma' and 'beta' are the weights and biases larned.

The script has many changes on the foundation of is ResNet50 by Francios Chollet, BigMoyan, Felix Yu
and other published results. I would like to thank all of them for the contributions. Make the necessary 
changes to adapt to the environment of TensorFlow 2.3, Keras 2.4.3, CUDA Toolkit 11.0, cuDNN 8.0.1 and 
CUDA 450.57. In addition, write the new code to replace the deprecated code. 

Environment: 

Ubuntu 18.04 
TensorFlow 2.3
Keras 2.4.3
CUDA Toolkit 11.0, 
cuDNN 8.0.1
CUDA 450.57. 

# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
"""


import os 
import warnings
import sys
sys.setrecursionlimit(3000)

import tensorflow as tf
from keras.layers import (Input, Dense, Conv2D, Flatten, Activation, MaxPooling2D, BatchNormalization, 
    ZeroPadding2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D)
from keras.layers import add 

from keras.models import Model
from keras import initializers
from keras.engine import Layer, InputSpec

from keras.engine.topology import get_source_inputs
from keras import backend as K
from imagenet_utils import _obtain_input_shape
# -from keras.utils.data_utils import get_file


# Set the visible environment 
os.environ["CUDA_DEVICE_ORDER"] ='0'

# Set the GPU limitation 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


os.environ["CUDA_DEVICE_ORDER"] ='0'
# -os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
WEIGHTS_PATH = '/home/mike/keras_dnn_models/resnet101_weights_tf.h5'


class Scale(Layer):
    # Custom Layer for ResNet used for BatchNormalization.
    """
    # Arguments
        axis: integer, axis along which to normalize in mode 0. 
        momentum: computate the exponential average of the mean and standard 
        deviation of the data, for feature-wise normalization.
        weights: List of 2 arrays, with shapes:`[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
        gamma_init: name of initialization function for scale parameter 
    """
    def __init__(self, weights=None, axis=-1, momentum=0.9, 
                 beta_init='zero', gamma_init='one', **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)
        self.gamma = K.variable(self.gamma_init(shape), 
                                name='{}_gamma'.format(self.name))
        self.beta = K.variable(self.beta_init(shape),
                               name='{}_beta'.format(self.name))
        self.train_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]
        out = K.reshape(self.gamma,broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


def identity_block(input_tensor, kernel_size, filters, stage, block):
    # The identity_block is the block that has no conv layer at shortcut
    """
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at the main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Return
        Return a intermidiary value 'x'
    """
    eps = 1.2e-5

    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1,1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1,1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(filters2, (kernel_size, kernel_size), 
               name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(filters3, (1,1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x,input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)

    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2)):
    # conv_block is the block that has a conv layer at shortcut
    """
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at the main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        Note that from stage 3, the first conv layer at main path is with
        subsample=(2,2). And the shortcut should have subsample=(2,2) as well
    # Return
        Return a intermidiary value 'x'
    """

    eps = 1.2e-5
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1,1), strides=strides, name=conv_name_base + '2a', 
               use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1,1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(filters2, (kernel_size,kernel_size), 
               name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(filters3, (1,1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1,1), strides=strides, name=conv_name_base + '1', 
                      use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, 
                                  name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)

    return x


def ResNet101(input_shape, num_classes, include_top, weights, 
              input_tensor=None, pooling=None):
    # Instantiates the ResNet152 architecture.
    """
    Arguments: 
        include_top: whether to include the FC layer at the top of the network.
        weights: `None` (random initialization), 'imagenet' or other path
        input_tensor: optional Keras tensor (output of `layers.Input()`)
        input_shape: tuple, `channels_last` data format) or `channels_first` 
        pooling: mode for feature extraction when `include_top` is `False`.
            - `None`: the output of model is the 4D tensor of the last conv layer 
            - `avg` means global average pooling and the output as a 2D tensor.
            - `max` means global max pooling will be applied.
        num_classes: specified if `include_top` is True
    Returns
        A Keras model instance.
    Raise
        ValueError: in case invalid argument for `weights` or input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and num_classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='data')
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape, name='data')
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    eps = 1.2e-5

    x = ZeroPadding2D((3,3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7,7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2), name='pool1')(x)

    x = conv_block(x, 3, [64,64,256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64,64,256], stage=2, block='b')
    x = identity_block(x, 3, [64,64,256], stage=2, block='c')

    x = conv_block(x, 3, [128,128,512], stage=3, block='a')
    for i in range(1, 3):
        x = identity_block(x, 3, [128,128,512], stage=3, block='b' + str(i))

    x = conv_block(x, 3, [256,256,1024], stage=4, block='a')
    for i in range(1, 23):
        x = identity_block(x, 3, [256,256,1024], stage=4, block='b' + str(i))

    x = conv_block(x, 3, [512,512,2048], stage=5, block='a')
    x = identity_block(x, 3, [512,512,2048], stage=5, block='b')
    x = identity_block(x, 3, [512,512,2048], stage=5, block='c')

    x = AveragePooling2D((7,7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(num_classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure the model considers any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Build the model.
    model = Model(inputs, x, name='resnet101')

    # Add 'by_name=True' into model.load_weights() for a correct shape. 
    if weights == 'imagenet':
        if include_top:
            weights_path = WEIGHTS_PATH
        else:
            weights_path = WEIGHTS_PATH_NO_TOP
        model.load_weights(weights_path, by_name=True)

    return model

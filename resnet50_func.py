#!/usr/bin/env python
# coding: utf-8

# resnet50_predict.py

"""
ResNet50 model for Keras.

Please pay more attention on the formal argument "x". To faciliate the process of parameter passing
during the function calls in the context, we select x to express the recursion that is the typical
mathematical usage.

Please remember that it is the TensorFlow realization with image_data_foramt = 'channels_last'. If
the env of Keras is 'channels_first', please change to the TensorFlow convention. While runnning
the script of predict.py, it shows the result as follows. 

Predicted: [[('n01930112', 'nematode', 0.13556267), ('n03207941', 'dishwasher', 0.032914065), 
('n03041632', 'cleaver', 0.02433419), ('n03804744', 'nail', 0.022761008), ('n02840245', 'binder', 
0.019043112)]]

The script has many changes on the foundation of is ResNet50 by Francios Chollet, BigMoyan and many 
other published results. I would like to thank all of them for the contributions. 

Make the the necessary changes to adapt to the environment of TensorFlow 2.3, Keras 2.4.3, CUDA Toolkit 
11.0, cuDNN 8.0.1 and CUDA 450.57. In addition, write the new lines of code to replace the deprecated 
code. 

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

import tensorflow as tf
from keras import layers
from keras.layers import Input, Conv2D, Dense, Activation, Flatten, MaxPooling2D, BatchNormalization, \
    GlobalMaxPooling2D, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


WEIGHTS_PATH = '/home/mike/keras_dnn_models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = '/home/mike/keras_dnn_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    # The identity block is the block that has no conv layer at shortcut.
    """
    # Arguments
        input_tensor: input tensor
        kernel_size: Being defualted 3, the kernel size of middle conv layer
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Return
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1,1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1,1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2)):
    # conv_block is the block that has a conv layer at shortcut
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1,1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1,1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1,1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    return x


def ResNet50(input_shape, num_classes, weights, include_top, 
             input_tensor=None, pooling=None):
    # Instantiates the ResNet50 architecture.
    """
    Arguments
        input_shape: Set (224,224,3) and height being larger than 197.
        include_top: whether to include the FC Layer at the top of the network.
        num_classes: specify 'include_top' is True for 1000.
        weights: None or imagenet (pre-training on ImageNet).
        input_tensor: Keras tensor, output of layers
        pooling: pooling mode for feature extraction while 'include_top' is False.
            - None: output of 4D tensor output of the last convolutional layer.
            - avg: global average pooling for the last conv layer with 2D tensor.
            - max: global max pooling being applied.
    Return
        A Keras model instance.
    """
    # Input() initizate a 3-D shape(weight,height,channels) into a 4-D tensor(batch, 
    # weight,height,channels). If no batch size, it is defaulted as None.
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=None, # -K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    x = ZeroPadding2D((3,3))(img_input)
    x = Conv2D(64, (7,7), strides=(2,2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)

    x = conv_block(x, 3, [64,64,256], stage=2, block='a', strides=(1,1))
    x = identity_block(x, 3, [64,64,256], stage=2, block='b')
    x = identity_block(x, 3, [64,64,256], stage=2, block='c')

    x = conv_block(x, 3, [128,128,512], stage=3, block='a')
    x = identity_block(x, 3, [128,128,512], stage=3, block='b')
    x = identity_block(x, 3, [128,128,512], stage=3, block='c')
    x = identity_block(x, 3, [128,128,512], stage=3, block='d')

    x = conv_block(x, 3, [256,256,1024], stage=4, block='a')
    x = identity_block(x, 3, [256,256,1024], stage=4, block='b')
    x = identity_block(x, 3, [256,256,1024], stage=4, block='c')
    x = identity_block(x, 3, [256,256,1024], stage=4, block='d')
    x = identity_block(x, 3, [256,256,1024], stage=4, block='e')
    x = identity_block(x, 3, [256,256,1024], stage=4, block='f')

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

    # Enable any potential predecessors of 'input_tensor'.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Build the model with both the inputs and outputs in the 4D tensors.
    model = Model(inputs, x, name='resnet50')

    # Add 'by_name=True' into model.load_weights() for a correct shape. 
    if weights == 'imagenet':
        if include_top:
            weights_path = WEIGHTS_PATH
        else:
            weights_path = WEIGHTS_PATH_NO_TOP
        model.load_weights(weights_path, by_name=True)

    return model

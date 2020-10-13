#!/usr/bin/env python
# coding: utf-8

# resnet50_v2_pred.py

"""
ResNet, ResNetV2, and ResNeXt models for Keras.

ResNet50V2 is the most accurate model for prection on any given image in the ResNet family. Even 
though it has the small total size of parameters about 25+ million, its accuracy is much highers. 
Other ResNet models such as ResNeXt101 has the 240+ million, but its prediction is much lowers.
Users need to run the script with the online weights downloading in the runtime or run the script 
after downloading the weights. 

Please remember that it is the TensorFlow realization with image_data_foramt = 'channels_last'. If
the env of Keras is 'channels_first', please change it according to the TensorFlow convention. The 
prediction is extremely than the inception v4 model. Therefore, we need to improve the method. 

$ python resnet50_v2_pred.py

The script has many modifications on the foundation of is ResNet Common by Francios Chollet. Make the 
necessary changes to adapt to the environment of TensorFlow 2.3, Keras 2.4.3, CUDA Toolkit 11.0, cuDNN 
8.0.1 and CUDA 450.57. In addition, write the new lines of code to replace the deprecated code. 

Environment: 

Ubuntu 18.04 
TensorFlow 2.3
Keras 2.4.3
CUDA Toolkit 11.0, 
cuDNN 8.0.1
CUDA 450.57. 

# Reference papers
- [Deep Residual Learning for Image Recognition]
  (https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
- [Identity Mappings in Deep Residual Networks]
  (https://arxiv.org/abs/1603.05027) (ECCV 2016)
- [Aggregated Residual Transformations for Deep Neural Networks]
  (https://arxiv.org/abs/1611.05431) (CVPR 2017)

# Reference implementations
- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/resnets.py)
- [Caffe ResNet]
  (https://github.com/KaimingHe/deep-residual-networks/tree/master/prototxt)
- [Torch ResNetV2]
  (https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua)
- [Torch ResNeXt]
  (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
"""


import os
import tensorflow as tf
import numpy as np
import warnings

import keras.backend as K
from keras import layers
from keras.layers import Add, Input,Dense, Conv2D, DepthwiseConv2D, Activation, Flatten, MaxPooling2D, \
    BatchNormalization, GlobalMaxPooling2D, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image

from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs


# Set up the GPU memory size to avoid the out-of-memory error
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


BASE_WEIGHTS_PATH = (
    'https://github.com/keras-team/keras-applications/releases/download/resnet/')
WEIGHTS_HASHES = {
    'resnet50': ('2cb95161c43110f7111970584f804107',
                 '4d473c1dd8becc155b73f8504c6f6626'),
    'resnet101': ('f1aeb4b969a6efcfb50fad2f0c20cfc5',
                  '88cf7a10940856eca736dc7b7e228a21'),
    'resnet152': ('100835be76be38e30d865e96f2aaae62',
                  'ee4c566cf9a93f14d82f913c2dc6dd0c'),
    'resnet50v2': ('3ef43a0b657b3be2300d5770ece849e0',
                   'fac2f116257151a9d068a22e544a4917'),
    'resnet101v2': ('6343647c601c52e1368623803854d971',
                    'c0ed64b8031c3730f411d2eb4eea35b5'),
    'resnet152v2': ('a49b44d1979771252814e80f8ec446f9',
                    'ed17cf2e0169df9d443503ef94b23b33'),
    'resnext50': ('67a5b30d522ed92f75a1f16eef299d1a',
                  '62527c363bdd9ec598bed41947b379fc'),
    'resnext101': ('34fb605428fcc7aa4d62f44404c11509',
                   '0f678c91647380debd923963594981b3')
}


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    # A residual block
    """
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    # Return
        Output tensor for the residual block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_relu')(x)

    x = Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_relu')(x)

    x = Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = Add(name=name + '_add')([shortcut, x])
    x = Activation('relu', name=name + '_out')(x)

    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    # A set of stacked residual blocks
    """
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.
    # Return
        Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, name=name + '_block1')

    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))

    return x


def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    # A residual block
    """
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    # Return
        Output tensor for the residual block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    preact = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_preact_bn')(x)
    preact = Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut is True:
        shortcut = Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(preact)
    else:
        shortcut = MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = Conv2D(filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_relu')(x)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = Conv2D(filters, kernel_size, strides=stride, use_bias=False, name=name + '_2_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_relu')(x)

    x = Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = Add(name=name + '_out')([shortcut, x])

    return x


def stack2(x, filters, blocks, stride1=2, name=None):
    # A set of stacked residual blocks.
    """
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.
    # Return
        Output tensor for the stacked blocks.
    """
    x = block2(x, filters, conv_shortcut=True, name=name + '_block1')

    for i in range(2, blocks):
        x = block2(x, filters, name=name + '_block' + str(i))

    x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))

    return x


def block3(x, filters, kernel_size=3, stride=1, groups=32, conv_shortcut=True, name=None):
    # A residual block.
    """
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        groups: default 32, group size for grouped convolution.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    # Return
        Output tensor for the residual block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = Conv2D((64 // groups) * filters, 1, strides=stride, use_bias=False, name=name + '_0_conv')(x)
        shortcut = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = Conv2D(filters, 1, use_bias=False, name=name + '_1_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_relu')(x)

    c = filters // groups

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = DepthwiseConv2D(kernel_size, strides=stride, depth_multiplier=c, use_bias=False, name=name + '_2_conv')(x)

    kernel = np.zeros((1, 1, filters * c, filters), dtype=np.float32)

    for i in range(filters):
        start = (i // c) * c * c + i % c
        end = start + c * c
        kernel[:, :, start:end:c, i] = 1.

    x = Conv2D(filters, 1, use_bias=False, trainable=False,
               kernel_initializer={'class_name': 'Constant', 'config': {'value': kernel}},
               name=name + '_2_gconv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_relu')(x)

    x = Conv2D((64 // groups) * filters, 1, use_bias=False, name=name + '_3_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = Add(name=name + '_add')([shortcut, x])
    x = Activation('relu', name=name + '_out')(x)

    return x


def stack3(x, filters, blocks, stride1=2, groups=32, name=None):
    # A set of stacked residual blocks.
    """
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        groups: default 32, group size for grouped convolution.
        name: string, stack label.
    # Return
        Output tensor for the stacked blocks.
    """
    x = block3(x, filters, stride=stride1, groups=groups, name=name + '_block1')

    for i in range(2, blocks + 1):
        x = block3(x, filters, groups=groups, conv_shortcut=False, name=name + '_block' + str(i))

    return x


def ResNet(stack_fn, preact, use_bias, model_name='resnet', include_top=True, weights='imagenet',
           input_tensor=None, input_shape=None, pooling=None, num_classes=1000, **kwargs):
    # Instantiates the ResNet, ResNetV2, and ResNeXt architecture.
    """
    # Arguments
        stack_fn: a function that returns output tensor for the stacked residual blocks.
        preact: whether to use pre-activation or not (True for ResNetV2, False for ResNet and ResNeXt).
        use_bias: whether use biases for conv layers or not (True for ResNet/ResNetV2, False for ResNeXt).
        model_name: string, model name.
        include_top: whether to include the FC layer at the top of the network.
        weights: `None` (random initialization), 'imagenet' or the path to any weights.
        input_tensor: optional Keras tensor (output of `layers.Input()`)
        input_shape: tuple, only to be specified if `include_top` is False.
        pooling: Optional mode for feature extraction when `include_top` is `False`.
            - `None`: the output of model is the 4D tensor of the last conv layer 
            - `avg` means global average pooling and the output as a 2D tensor.
            - `max` means global max pooling will be applied.
        num_classes: specified if `include_top` is True
        num_classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights` or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and num_classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    x = ZeroPadding2D(padding=((3,3), (3,3)), name='conv1_pad')(img_input)
    x = Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    if preact is False:
        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
        x = Activation('relu', name='conv1_relu')(x)

    x = ZeroPadding2D(padding=((1,1), (1,1)), name='pool1_pad')(x)
    x = MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    if preact is True:
        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='post_bn')(x)
        x = Activation('relu', name='post_relu')(x)

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(num_classes, activation='softmax', name='probs')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure the model considers any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Build the model.
    model = Model(inputs, x, name=model_name)

    # Load weights.
    if (weights == 'imagenet') and (model_name in WEIGHTS_HASHES):
        if include_top:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels.h5'
            file_hash = WEIGHTS_HASHES[model_name][0]
        else:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
            file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = get_file(file_name,
                                BASE_WEIGHTS_PATH + file_name,
                                cache_subdir='models',
                                file_hash=file_hash)
        by_name = True if 'resnext' in model_name else False
        model.load_weights(weights_path, by_name=True)
    elif weights is not None:
        model.load_weights(weights)

    return model 


def ResNet50(include_top=True, weights='imagenet', input_tensor=None, 
             input_shape=None, pooling=None, num_classes=1000, **kwargs):

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 4, name='conv3')
        x = stack1(x, 256, 6, name='conv4')
        x = stack1(x, 512, 3, name='conv5')

        return x

    return ResNet(stack_fn, False, True, 'resnet50', include_top, weights,
                  input_tensor, input_shape, pooling, num_classes, **kwargs)


def ResNet101(include_top=True, weights='imagenet', input_tensor=None, 
              input_shape=None, pooling=None, num_classes=1000, **kwargs):

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 4, name='conv3')
        x = stack1(x, 256, 23, name='conv4')
        x = stack1(x, 512, 3, name='conv5')

        return x

    return ResNet(stack_fn, False, True, 'resnet101', include_top, weights,
                  input_tensor, input_shape, pooling, num_classes, **kwargs)


def ResNet152(include_top=True, weights='imagenet', input_tensor=None,
              input_shape=None, pooling=None, num_classes=1000, **kwargs):

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 8, name='conv3')
        x = stack1(x, 256, 36, name='conv4')
        x = stack1(x, 512, 3, name='conv5')

        return x

    return ResNet(stack_fn, False, True, 'resnet152', include_top, weights,
                  input_tensor, input_shape, pooling, num_classes, **kwargs)


def ResNet50V2(include_top=True, weights='imagenet', input_tensor=None,
               input_shape=None, pooling=None, num_classes=1000, **kwargs):

    def stack_fn(x):
        x = stack2(x, 64, 3, name='conv2')
        x = stack2(x, 128, 4, name='conv3')
        x = stack2(x, 256, 6, name='conv4')
        x = stack2(x, 512, 3, stride1=1, name='conv5')

        return x

    return ResNet(stack_fn, True, True, 'resnet50v2', include_top, weights,
                  input_tensor, input_shape, pooling, num_classes, **kwargs)


def ResNet101V2(include_top=True, weights='imagenet', input_tensor=None,
                input_shape=None, pooling=None, num_classes=1000, **kwargs):

    def stack_fn(x):
        x = stack2(x, 64, 3, name='conv2')
        x = stack2(x, 128, 4, name='conv3')
        x = stack2(x, 256, 23, name='conv4')
        x = stack2(x, 512, 3, stride1=1, name='conv5')

        return x

    return ResNet(stack_fn, True, True, 'resnet101v2', include_top, weights,
                  input_tensor, input_shape, pooling, num_classes, **kwargs)


def ResNet152V2(include_top=True, weights='imagenet',input_tensor=None,
                input_shape=None, pooling=None, num_classes=1000, **kwargs):

    def stack_fn(x):
        x = stack2(x, 64, 3, name='conv2')
        x = stack2(x, 128, 8, name='conv3')
        x = stack2(x, 256, 36, name='conv4')
        x = stack2(x, 512, 3, stride1=1, name='conv5')

        return x

    return ResNet(stack_fn, True, True, 'resnet152v2', include_top, weights,
                  input_tensor, input_shape, pooling, num_classes, **kwargs)


def ResNeXt50(include_top=True, weights='imagenet', input_tensor=None,
              input_shape=None, pooling=None, num_classes=1000, **kwargs):

    def stack_fn(x):
        x = stack3(x, 128, 3, stride1=1, name='conv2')
        x = stack3(x, 256, 4, name='conv3')
        x = stack3(x, 512, 6, name='conv4')
        x = stack3(x, 1024, 3, name='conv5')

        return x

    return ResNet(stack_fn, False, False, 'resnext50', include_top, weights,
                  input_tensor, input_shape, pooling, num_classes, **kwargs)


def ResNeXt101(include_top=True, weights='imagenet', input_tensor=None,
               input_shape=None, pooling=None, num_classes=1000, **kwargs):

    def stack_fn(x):
        x = stack3(x, 128, 3, stride1=1, name='conv2')
        x = stack3(x, 256, 4, name='conv3')
        x = stack3(x, 512, 23, name='conv4')
        x = stack3(x, 1024, 3, name='conv5')

        return x

    return ResNet(stack_fn, False, False, 'resnext101', include_top, weights,
                  input_tensor, input_shape, pooling, num_classes, **kwargs)


setattr(ResNet50, '__doc__', ResNet.__doc__)
setattr(ResNet101, '__doc__', ResNet.__doc__)
setattr(ResNet152, '__doc__', ResNet.__doc__)
setattr(ResNet50V2, '__doc__', ResNet.__doc__)
setattr(ResNet101V2, '__doc__', ResNet.__doc__)
setattr(ResNet152V2, '__doc__', ResNet.__doc__)
setattr(ResNeXt50, '__doc__', ResNet.__doc__)
setattr(ResNeXt101, '__doc__', ResNet.__doc__)


def preprocess_input(x):
    # Process any given image
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    output = np.multiply(x, 2.0)

    return output


if __name__ == '__main__':

    model = ResNet50V2(include_top=True, weights='imagenet', input_tensor=None, 
                       input_shape=(224,224,3), pooling=None, num_classes=1000)

    model.summary()

    img_path = '/home/mike/Documents/keras_resnet_common/images/plane.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    output = preprocess_input(img)

    print('Input image shape:', output.shape)

    preds = model.predict(output)
    print('Predicted:', decode_predictions(preds))

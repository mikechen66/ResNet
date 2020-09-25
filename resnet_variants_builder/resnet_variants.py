#!/usr/bin/env python
# coding: utf-8

# resnet_variants.py
 
"""
The ResNet variants includes 18, 34, 50, 101, 152 levels network with calling ResnetBuilder within 
resnet_builder.py. Since the 1000-levels ResNet is not economical in the computation, so it not
include the huge level ResNet. 

Please pay more attention on the formal argument "x". To faciliate the process of parameter passing
during the function calls in the context, we select x to express the recursion that is the typical
mathematical usage.

Please remember it is the TensorFlow realization with image_data_foramt = 'channels_last'. Please 
change to the TensorFlow convention.

Make the the necessary changes to adapt to the environment of TensorFlow 2.3, Keras 2.4.3, CUDA Toolkit 
11.0, cuDNN 8.0.1 and CUDA 450.57. In addition, write the new lines of code to replace the deprecated 
code. 

It is simplified script based lamsade.dauphine. I would like to thank all of them for the 
contributions. The script has many changes on the foundation of is ResNet50 by Francios Chollet, Kent 
Sommers, lamsade.dauphine and many other published results. I would like to thank all of them for the 
contributions. 

Environment: 

Ubuntu 18.04 
TensorFlow 2.3
Keras 2.4.3
CUDA Toolkit 11.0, 
cuDNN 8.0.1
CUDA 450.57.
"""


from keras import backend as K
from keras.models import Model
from keras.applications import ResNet50, InceptionV3
from keras.layers import Input, Conv2D, Activation, Flatten, Dropout, MaxPooling2D, ZeroPadding2D, \
    AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization

from keras.layers import Dense
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
# -from keras_applications.imagenet_utils import _obtain_input_shape
from imagenet_utils import _obtain_input_shape
from resnet50_func import identity_block, conv_block
from keras.utils import layer_utils
from resnet_build import ResnetBuilder


WEIGHTS_PATH = '/home/mic/keras_dnn_models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = '/home/mic/keras_dnn_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def resnet_trained(n_retrain_layers=0):
    K.set_image_data_format('channels_last')
    base_model = ResNet50(include_top=False, input_shape=(224,224,3))
    features = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=features)
    model = _set_n_retrain(model, n_retrain_layers)

    return model


def inception(n_retrain_layers=0):
    K.set_image_data_format('channels_last')
    base_model = InceptionV3(include_top=False, input_shape=(224,224,3))
    features = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=features)
    model = _set_n_retrain(model, n_retrain_layers)

    return model


def resnet_trained_2(n_retrain_layers=0):
    K.set_image_data_format('channels_last')
    base_model = ResNet50(include_top=False, input_shape=(224,224,3))
    features = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=features)
    model = _set_n_retrain(model, n_retrain_layers, reinit=True)

    return model


def empty_resnet():
    K.set_image_data_format('channels_last')
    base_model = ResNet50(weights=None, include_top=False, input_shape=(224,224,3))
    features = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=features)

    return model


def resnet18():
    return ResnetBuilder.build_resnet_18((3,224,224), 25)


def resnet34():
    return ResnetBuilder.build_resnet_34((3,224,224), 25)


def resnet101():
    return ResnetBuilder.build_resnet_101((3,224,224), 25)


def resnet152():
    return ResnetBuilder.build_resnet_152((3,224,224), 25)


def custom_resnet(n=0, drop_rate=0):

    # Input() initizate a 3D shape(weight,height,channels) into a 4D tensor(batch, 
    # weight,height,channels). If no batch size, it is defaulted as None.
    img_input = Input(shape=(224,224,3))

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3,3))(img_input)
    x = Conv2D(64, (7,7), strides=(2,2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)

    x = conv_block(x, 3, [64,64,256], stage=2, block='a', strides=(1,1))
    x = identity_block(x, 3, [64,64,256], stage=2, block='b')
    x = identity_block(x, 3, [64,64,256], stage=2, block='c')


    x = conv_block(x, 3, [128,128,512], stage=3, block='a')
    x = identity_block(x, 3, [128,128,512], stage=3, block='b')
    x = identity_block(x, 3, [128,128,512], stage=3, block='c')
    x = identity_block(x, 3, [128,128,512], stage=3, block='d')

    # Add the first funcion of Dropout()
    x = Dropout(drop_rate)(x)

    x = conv_block(x, 3, [256,256,1024], stage=4, block='a')
    x = identity_block(x, 3, [256,256,1024], stage=4, block='b')
    x = identity_block(x, 3, [256,256,1024], stage=4, block='c')
    x = identity_block(x, 3, [256,256,1024], stage=4, block='d')
    x = identity_block(x, 3, [256,256,1024], stage=4, block='e')
    x = identity_block(x, 3, [256,256,1024], stage=4, block='f')

    # Add the second function of Dropout()
    x = Dropout(drop_rate)(x)

    # Remove the three statements from ResNet50
    # -x = conv_block(x, 3, [512,512,2048], stage=5, block='a')
    # -x = identity_block(x, 3, [512,512,2048], stage=5, block='b')
    # -x = identity_block(x, 3, [512,512,2048], stage=5, block='c')

    x = AveragePooling2D((7,7), name='avg_pool')(x)

    x = Flatten()(x)
    # x = Dense(25, activation='softmax', name='fc1000')(x)
    x = Dense(num_classes, activation='softmax', name='fc1000')(x)

    # Ensure the model consider any potential predecessors of `input_tensor`.
    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    weights_path = WEIGHTS_PATH_NO_TOP
    model.load_weights(weights_path, by_name=True)

    # Add the stamement of split_value 
    split_value = True # len(model.layers) + 1 - n
    for layer in model.layers[:split_value]:
        layer.trainable = False
    for layer in model.layers[split_value:]:
        layer.trainable = True

    return model


def resnet_dropout(include_top=False, weights='imagenet', input_tensor = None, pooling='avg', 
                   input_shape=(224,224,3), classes=25, drop_rate=0., n_retrain_layers=0):

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Input() initizates a 3D shape(weight,height,channels) into a 4D tensor(batch,weight,height,
    # channels). If no batch size, it is defaulted as None.
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3,3))(img_input)
    x = Conv2D(64, (7,7), strides=(2,2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)

    x = conv_block(x, 3, [64,64,256], stage=2, block='a', strides=(1, 1))
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
    x = Dropout(drop_rate)(x)
    x = identity_block(x, 3, [512,512,2048], stage=5, block='c')

    x = AveragePooling2D((7,7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure the model to consider any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = WEIGHTS_PATH
        else:
            weights_path = WEIGHTS_PATH_NO_TOP
        model.load_weights(weights_path)

    split_value = len(model.layers) + 1 - n_retrain_layers
    for layer in model.layers[:split_value]:
        layer.trainable = False
    for layer in model.layers[split_value:]:
        layer.trainable = True

    return model


def _get_weighted_layers(model):
    res=[]
    for layer in model.layers:
        if len(layer.get_weights()) != 0:
            res.append(layer.name)

    return res


def _set_n_retrain(model, n, reinit=False):
    w_layers = _get_weighted_layers(model)

    if reinit:
        empty_model = empty_resnet()

    if n > len(w_layers):
        n == len(w_layers)

    if n > 0:
        if reinit:
            for layer, layer_empty in zip(model.layers, empty_model.layers):
                if layer.name in w_layers[-n:]:
                    layer.trainable = True
                    w = layer_empty.get_weights()
                    layer.set_weights(w)
                else:
                    layer.trainable = False
        else :
            for layer in model.layers:
                if layer.name in w_layers[-n:]:
                    layer.trainable = True
                else:
                    layer.trainable = False

    else :
        for layer in model.layers:
            layer.trainable = False

    return model


if __name__ == '__main__':

    model = resnet_trained(1)

    model.summary()
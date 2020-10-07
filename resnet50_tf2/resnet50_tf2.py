# resnet50_tf2.py

"""
ResNet50 model for Keras.

Please pay more attention on the formal argument "x". To faciliate the process of parameter passing
during the function calls in the context, we select x to express the recursion that is the typical
mathematical usage. Remember it is the TensorFlow realization with image_data_foramt = 'channels_last'. 
If the env of Keras is 'channels_first', please change it according to the TensorFlow convention. 
Please run the script with the command as follows. 

$ python resnet50_tf2.py

Even adopting the validation_utils of imageNet and changing prediction methods in predict_val.py, its 
correctedness is extremely lower than Inception v3 becuase the residual layer increases the "raw" data 
greatly. So it is subject to the brute force computing, i.e., updating the moving average from 100 to 
1000 epochs before converging to the "real" mean and variance. That's why ResNet predicts a wrong result 
in the early stages. Please verify it by forcing the BatchNorm Layer to run in the "Training mode".

The total size of parameters is about 20 million that is in the range of the official ResNet network. 
User can change the total parameters with resizing the kernel size. 

The script has many changes on the foundation of is ResNet50 by Francios Chollet, BigMoyan,and other 
published results. I would like to thank all of them for the contributions. Make the necessary changes 
to adapt to the environment of TensorFlow 2.3, Keras 2.4.3, CUDA Toolkit 11.0, cuDNN 8.0.1 and CUDA 
450.57. In addition, write the new code to replace the deprecated code. 

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
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from keras.initializers import he_normal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Flatten, Activation, BatchNormalization, \
    ZeroPadding2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Give the path of the weights 
WEIGHTS_PATH = '/home/mike/keras_dnn_models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = '/home/mike/keras_dnn_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def iden_block(x, kernel_size, filters, stage, block):
	# Implemente the identity block
    """
    Arguments:
    x -- input tensor of shape
    kernel_size -- integer, specifying the shape of the middle Conv's window for the main path
    filters -- a list of of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    Return:
    x -- output of the identity block
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    [F1, F2, F3] = filters
    
    # Save the input value that is needed later to add back to the main path. 
    x_shortcut = x
    
    # First component of main path
    x = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid', name=conv_name_base + '2a', 
    	       kernel_initializer=he_normal(seed=0))(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    
    # Second component of main path (≈3 lines)
    x = Conv2D(filters=F2, kernel_size=kernel_size, strides=(1,1), padding='same', name=conv_name_base + '2b', 
    	       kernel_initializer=he_normal(seed=0))(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # Third component of main path (≈2 lines)
    x = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=conv_name_base + '2c', 
    	       kernel_initializer=he_normal(seed=0))(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    # Final step: add shortcut value to the main path, and pass it through a RELU activation (≈2 lines)
    x = layers.add([x, x_shortcut])
    x = Activation('relu')(x)
    
    return x


def conv_block(x, kernel_size, filters, stage, block, s=2):
	# Define the convolutional block
    """
    Arguments:
    x -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- a list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- integer, specifying the stride to be used
    Return:
    x -- a 4D tensor output of the convolutional block
    """
    
    # Define the name of base
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    [F1, F2, F3] = filters
    
    # Save the input value
    x_shortcut = x

    # First component of main path 
    x = Conv2D(F1, kernel_size=(1,1), strides=(s,s), name=conv_name_base + '2a', padding='valid', 
    	       kernel_initializer=he_normal(seed=0))(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # Second component of main path (≈3 lines)
    x = Conv2D(F2, kernel_size=kernel_size, strides=(1,1), name=conv_name_base + '2b', padding='same', 
    	       kernel_initializer=he_normal(seed=0))(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # Third component of main path (≈2 lines)
    x = Conv2D(F3, kernel_size=(1,1), strides=(1,1), name=conv_name_base + '2c', padding='valid', 
    	       kernel_initializer=he_normal(seed=0))(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    # Shortcut path 
    x_shortcut = Conv2D(F3, kernel_size=(1,1), strides=(s,s), name=conv_name_base + '1', padding='valid', 
    	                kernel_initializer=he_normal(seed=0))(x_shortcut)
    x_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(x_shortcut)

    # Final step: Add shortcut value to the main path and pass it through a RELU activation (≈2 lines)
    x = layers.add([x, x_shortcut])
    x = Activation('relu')(x)
    
    return x


def ResNet50(input_shape, num_classes, include_top, weights):
	# Define the nerwork of ResNet50
    """ 
    Arguments:
    input_shape: Set (224,224,3) and height being larger than 197.
    include_top: whether to include the FC Layer at the top of the network.
    num_classes: specify 'include_top' is True for 1000.
    weights: None or imagenet (pre-training on ImageNet).
    Return: 
    A Keras model instance.
    """

    # Initizate a 3D shape(weight,height,channels) into a 4D tensor(batch, weight, height, channels). 
    # If no batch size, it is defaulted as None.
    inputs = Input(input_shape)

    # Zero-Padding
    x = ZeroPadding2D((3,3))(inputs)
    
    # Stage 1
    x = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), name='conv1', kernel_initializer=he_normal(seed=0))(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

    # Stage 2
    x = conv_block(x, kernel_size=(3,3), filters=[64,64,256], stage=2, block='a', s=1)
    x = iden_block(x, kernel_size=(3,3), filters=[64,64,256], stage=2, block='b')
    x = iden_block(x, kernel_size=(3,3), filters=[64,64,256], stage=2, block='c')

    # Stage 3 (≈4 lines)
    x = conv_block(x, kernel_size=(3,3), filters=[128,128,512], stage=3, block='a', s=2)
    x = iden_block(x, kernel_size=(3,3), filters=[128,128,512], stage=3, block='b')
    x = iden_block(x, kernel_size=(3,3), filters=[128,128,512], stage=3, block='c')
    x = iden_block(x, kernel_size=(3,3), filters=[128,128,512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    x = conv_block(x, kernel_size=(3,3), filters=[256,256,1024], block='a', stage=4, s=2)
    x = iden_block(x, kernel_size=(3,3), filters=[256,256,1024], block='b', stage=4)
    x = iden_block(x, kernel_size=(3,3), filters=[256,256,1024], block='c', stage=4)
    x = iden_block(x, kernel_size=(3,3), filters=[256,256,1024], block='d', stage=4)
    x = iden_block(x, kernel_size=(3,3), filters=[256,256,1024], block='e', stage=4)
    x = iden_block(x, kernel_size=(3,3), filters=[256,256,1024], block='f', stage=4)

    # Stage 5 (≈3 lines)
    x = conv_block(x, kernel_size=(3,3), filters=[512,512,2048], stage=5, block='a', s=2)
    x = iden_block(x, kernel_size=(3,3), filters=[256,256,2048], stage=5, block='b')
    x = iden_block(x, kernel_size=(3,3), filters=[256,256,2048], stage=5, block='c')

    # AVGPOOL (≈1 line)
    x = AveragePooling2D(pool_size=(7,7))(x)
    
    if include_top:
        x = Flatten()(x)
        x = Dense(num_classes, activation='softmax', name='fc1000' + str(num_classes), kernel_initializer = he_normal(seed=0))(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
       
    # Build the model by the 4D tensors 
    model = Model(inputs=inputs, outputs=x, name='ResNet50')

    # Add 'by_name=True' into model.load_weights() for a correct shape. 
    if weights == 'imagenet':
        if include_top:
            weights_path = WEIGHTS_PATH
        else:
            weights_path = WEIGHTS_PATH_NO_TOP
        model.load_weights(weights_path, by_name=True)

    return model


if __name__ == '__main__':

    input_shape = (299, 299, 3)
    num_classes = 1000
    include_top=True
    weights='imagenet'

    model = ResNet50(input_shape, num_classes, weights, include_top)

    model.summary()

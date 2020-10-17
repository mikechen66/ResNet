#!/usr/bin/env python
# coding: utf-8

# predict_resnet50v2.py

"""
ResNet50 model for Keras.

Please remember that it is the TensorFlow realization with image_data_foramt = 'channels_last'. If
the env of Keras is 'channels_first', please change it  according to the TensorFlow convention. The 
prediction is extremely than the inception v4 model. Therefore, we need to improve the method.  

$ python predict_101.py

# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

The script has many modifications on the foundation of is ResNet Common by Francios Chollet. Make the 
necessary changes to adapt to the environment of TensorFlow 2.3, Keras 2.4.3, CUDA Toolkit 11.0, cuDNN 
8.0.1 and CUDA 450.57. In addition, write the new lines of code to replace the deprecated 
code. 

Environment: 

Ubuntu 18.04 
TensorFlow 2.3
Keras 2.4.3
CUDA Toolkit 11.0, 
cuDNN 8.0.1
CUDA 450.57. 
"""


import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from resnet_common import ResNet101


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


def preprocess_input(x):
    # Process any given image
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    output = np.multiply(x, 2.0)

    return output


if __name__ == '__main__':

    model = ResNet101(include_top=True, weights='imagenet', input_tensor=None, 
                       input_shape=(224,224,3), pooling=None, num_classes=1000)

    model.summary()

    img_path = '/home/mike/Documents/keras_resnet_common/images/cat.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    output = preprocess_input(img)

    print('Input image shape:', output.shape)

    preds = model.predict(output)
    print('Predicted:', decode_predictions(preds))
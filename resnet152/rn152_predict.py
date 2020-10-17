#!/usr/bin/env python
# coding: utf-8

# rn152_predict.py

"""
Provide the client for the ResNet152 model for Keras.

Please remember it is the TensorFlow realization with image_data_foramt = 'channels_last'.
The prediction is extremely lowee than the inception v4 model. So we need to improve the 
training method.  

$ python rn152_predict.py

Make the necessary changes to adapt to the environment of TensorFlow 2.3, Keras 2.4.3, CUDA 
Toolkit 11.0, cuDNN 8.0.1 and CUDA 450.57. In addition, update the lines of code to replace 
the deprecated code. 

# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
"""


import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import decode_predictions 
# -from keras.applications.resnet50 import ResNet50
# -from resnet101 import ResNet101
from resnet152 import ResNet152


def preprocess_input(x):
    # Process any given image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    output = np.multiply(x, 2.0)

    return output


if __name__ == '__main__':

    input_shape = (224,224,3)
    num_classes = 1000
    include_top = True
    weights='imagenet'

    model = ResNet152(input_shape, num_classes, include_top, weights)
    model.summary()

    img_path = '/home/mike/Documents/keras_resnet_v1/images/cat.jpg'
    img = image.load_img(img_path, target_size=(224,224))
    output = preprocess_input(img)

    print('Input image shape:', output.shape)

    preds = model.predict(output)
    print('Predicted:', decode_predictions(preds))

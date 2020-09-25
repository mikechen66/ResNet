#!/usr/bin/env python
# coding: utf-8

# resnet50_predict.py

"""
ResNet50 model for Keras.

Please remember that it is the TensorFlow realization with image_data_foramt = 'channels_last'. If
the env of Keras is 'channels_first', please change it  according to the TensorFlow convention. 

$ python predict.py

Predicted: [[('n01930112', 'nematode', 0.13556267), ('n03207941', 'dishwasher', 0.032914065), 
('n03041632', 'cleaver', 0.02433419), ('n03804744', 'nail', 0.022761008), ('n02840245', 'binder', 
0.019043112)]]

Even adopting the validation_utils of imageNet and change the prediction method in predict_val.py, the 
prediction is extremely than the inception v4 model. So we need to improve the ResNet training.

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


import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions  #preprocess_input
from resnet50_func import ResNet50


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


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
    weights='imagenet'
    include_top = True 

    model = ResNet50(input_shape, num_classes, weights, include_top)
    model.summary()

    img_path = '/home/mike/Documents/keras_resnet_v1/plane.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    output = preprocess_input(img)

    print('Input image shape:', output.shape)

    preds = model.predict(output)
    print('Predicted:', decode_predictions(preds))

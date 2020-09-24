#!/usr/bin/env python
# coding: utf-8

# predict.py

"""
ResNet50 model for Keras.

Please pay more attention on the formal argument "x". To faciliate the process of parameter passing
during the function calls in the context, we select x to express the recursion that is the typical
mathematical usage. 

We adopt the validation_utils of imageNet in the script and change the prediction method. But the 
prediction is extremely than the inception v4 model.Therefore, we need to improve the ResNet training.

$ python predict_val.py

Predicted: [[('n01930112', 'nematode', 0.13556267), ('n03207941', 'dishwasher', 0.032914065), 
('n03041632', 'cleaver', 0.02433419), ('n03804744', 'nail', 0.022761008), ('n02840245', 'binder', 
0.019043112)]]

The script has many changes on the foundation of is ResNet50 by Francios Chollet, BigMoyan and many 
other published results. I would like to thank all of them for the contributions. 

Make the necessary changes to adapt to the environment of TensorFlow 2.3, Keras 2.4.3, CUDA Toolkit 
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
-[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
"""


import numpy as np
import tensorflow as tf
from keras.preprocessing import image
# -from keras.applications.imagenet_utils import decode_predictions
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

    # Open the class label dictionary(that is a human readable label-given ID)
    classes = eval(open('/home/mike/Documents/keras_resnet_v1/validation_utils/class_names.txt', 'r').read())

    # Run the prediction on the given image
    preds = model.predict(output)
    print("Class is: " + classes[np.argmax(preds)-1])
    print("Certainty is: " + str(preds[0][np.argmax(preds)]))
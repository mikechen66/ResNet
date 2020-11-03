#!/usr/bin/env python
# coding: utf-8

import sys

sys.path.insert(0, "/home/mic/Documents/stacked_hourglass_tf2/src/data_gen/")
sys.path.insert(0, "/home/mic/Documents/stacked_hourglass_tf2/src/net/")

import argparse
import os
import tensorflow as tf

from keras import backend as k
from hourglass import HourglassNet
import tensorflow as tf 

"""
# Set up the GPU memory size to avoid the out-of-memory error
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 5GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)"""

# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuID", default=0, type=int, help='gpu id')
    parser.add_argument("--mobile", default=False, help="use depthwise conv in hourglass'")
    parser.add_argument("--batch_size", default=8, type=int, help='batch size for training')
    parser.add_argument("--model_path", help='path to store trained model')
    parser.add_argument("--num_stack", default=2, type=int, help='num of stacks')
    parser.add_argument("--epochs", default=20, type=int, help="number of traning epochs")
    parser.add_argument("--resume", default=False, type=bool, help="resume training or not")
    parser.add_argument("--resume_model", help="start point to retrain")
    parser.add_argument("--resume_model_json", help="model json")
    parser.add_argument("--init_epoch", type=int, help="epoch to resume")
    parser.add_argument("--tiny", default=False, type=bool, help="tiny network for speed, inres=[192x128], channel=128")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

    """
    # TensorFlow wizardry
    # -config = tf.ConfigProto()
    config = tf.compat.v1.ConfigProto()

    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    # -config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    # Create a session with the above options specified.
    # AttributeError: module 'keras.backend' has no attribute 'tensorflow_backend
    # -k.tensorflow_backend.set_session(tf.Session(config=config))
    # AttributeError: module 'keras.backend' has no attribute 'set_session'
    # -k.set_session(tf.Session(config=config))
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))"""

    if args.tiny:
        xnet = HourglassNet(num_classes=16, num_stacks=args.num_stack, num_channels=128, inres=(192, 192),
                            outres=(48, 48))
    else:
        xnet = HourglassNet(num_classes=16, num_stacks=args.num_stack, num_channels=256, inres=(256, 256),
                            outres=(64, 64))

    if args.resume:
        xnet.resume_train(batch_size=args.batch_size, model_json=args.resume_model_json,
                          model_weights=args.resume_model,
                          init_epoch=args.init_epoch, epochs=epochs)

    else:
        xnet.build_model(mobile=args.mobile, show=True)
        # NameError: name 'epochs' is not defined
        # -xnet.train(epochs=epochs, model_path=model_path, batch_size=batch_size)
        xnet.train(epochs=args.epochs, model_path=args.model_path, batch_size=args.batch_size)

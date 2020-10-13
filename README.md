# ResNet18-152

Residual Network (ResNet) is a CNN architecture which was designed to enable hundreds or thousands 
of convolutional layers. It includes both the identity shortcut and the convolutional blocks that 
elimiate the gradient descent vanlshing. All layers that are expanded and the residual parts of the 
network explore more effective feature space of the source image. 

Only ResNet50V2 predicts a much higher accuracy(82%) while other ResNet models are worse rate(16%~60%). 
It is caused by the residual. While momentum is defaulted as 0.99, moving_mean and moving_variance are 
initialized to 0 and 1 respecfully in the BatchNorm. So it is necessary to update the moving average
from 100 to 1000 epochs before converging to the "real" mean and variance. That's why the prediction 
was wrong in the early stages with ResNet.

The script has many changes on the foundation of is ResNet50 by Francios Chollet, BigMoyan and many 
other published results. I would like to thank all of them for the contributions. ake the necessary 
changes to adapt to the environment of TensorFlow 2.3, Keras 2.4.3, CUDA Toolkit 11.0, cuDNN 8.0.1 
and CUDA 450.57. In addition, write the new lines of code to replace the deprecated. 

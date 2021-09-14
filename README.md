Residual Networks
============

In theory, very deep networks can represent very complex functions; but in practice, they are hard to train. Residual Networks., allow you to train much deeper networks than were previously practically feasible. introduced by [ResNet](https://arxiv.org/pdf/1512.03385.pdf)

# Building a Residual Network
In ResNets, a `shortcut` or a `skip connection` allows the gradient to be directly backpropagated to earlier layers

<br>Two main types of blocks are used in a ResNet, depending mainly on whether the input/output dimensions are same or different. You are going to implement both of them.

## The identity block
The identity block is the standard block used in ResNets, and corresponds to the case where the input activation has the same dimension as the output activation . To flesh out the different steps of what happens in a ResNet's identity block, here is an alternative diagram showing the individual steps:<br>
![image](/image/Identityblock.png)
<br>The upper path is the "shortcut path." The lower path is the "main path." In this diagram, we have also made explicit the CONV2D and ReLU steps in each layer. To speed up training we have also added a BatchNorm step. 


## The convolutional block
the ResNet "convolutional block" is the other type of block. You can use this type of block when the input and output dimensions don't match up. The difference with the identity block is that there is a CONV2D layer in the shortcut path:
![image](/image/convolutionalblock.jpg)
<br>The CONV2D layer in the shortcut path is used to resize the input to a different dimension, so that the dimensions match up in the final addition needed to add the shortcut value back to the main path. For example, to reduce the activation dimensions's height and width by a factor of 2, you can use a 1x1 convolution with a stride of 2. The CONV2D layer on the shortcut path does not use any non-linear activation function. Its main role is to just apply a (learned) linear function that reduces the dimension of the input, so that the dimensions match up for the later addition step.

## Building the ResNet model (50 layers)
The following figure describes in detail the architecture of this neural network:
![](/image/model.png)
<br>The details of this ResNet-50 model are:

* Zero-padding pads the input with a pad of (3,3)
* Stage 1:
  * The 2D Convolution has 64 filters of shape (7,7) and uses a stride of (2,2). Its name is "conv1".
  * BatchNorm is applied to the channels axis of the input.
  * MaxPooling uses a (3,3) window and a (2,2) stride.
* Stage 2:
  * The convolutional block uses three set of filters of size [64,64,256], "f" is 3, "s" is 1 and the block is "a".
  * The 2 identity blocks use three set of filters of size [64,64,256], "f" is 3 and the blocks are "b" and "c".
* Stage 3:
  * The convolutional block uses three set of filters of size [128,128,512], "f" is 3, "s" is 2 and the block is "a".
  * The 3 identity blocks use three set of filters of size [128,128,512], "f" is 3 and the blocks are "b", "c" and "d".
* Stage 4:
  * The convolutional block uses three set of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and the block is "a".
  * The 5 identity blocks use three set of filters of size [256, 256, 1024], "f" is 3 and the blocks are "b", "c", "d", "e" and "f".
* Stage 5:
 * The convolutional block uses three set of filters of size [512, 512, 2048], "f" is 3, "s" is 2 and the block is "a".
 * The 2 identity blocks use three set of filters of size [512, 512, 2048], "f" is 3 and the blocks are "b" and "c".
* The 2D Average Pooling uses a window of shape (2,2) and its name is "avg_pool".
* The flatten doesn't have any hyperparameters or name.
* The Fully Connected (Dense) layer reduces its input to the number of classes using a softmax activation. 

## Conclusion
the conclusion of the pretrained Resnet50:
![image](/image/testaccuracy.png)


# References
This notebook presents the ResNet algorithm due to He et al. (2015). The implementation here also took significant inspiration and follows the structure given in the github repository of Francois Chollet:

<br>Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun - [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/pdf/1512.03385.pdf)
<br>Francois Chollet's github repository: [Resnet50.py](https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py)

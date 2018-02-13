# in4155-2017-fabian-hainzl

An implementation of "Visualizing and Understanding Convolutional Networks" by Matthew D. Zeiler and Rob Fergus [https://www.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf] in Keras/Tensorflow.

The pre-trained AlexNet is from:
https://github.com/heuritech/convnets-keras

To understand: Check out Poster of this project or the original paper.

To run: Downloatd the ILSVRC2012 validation set from http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads (6.3GB) in folder "ILSVRC2012_img_val", run activations.py (this may take a considerable amount of time) and then run deconvolution.py with a certain layer and filter as arguments to project the maxmial activation of a filter back to image space. 


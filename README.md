# in4155-2017-fabian-hainzl

An implementation of "Visualizing and Understanding
Convolutional Networks" by Matthew D. Zeiler and Rob Fergus [https://www.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf] in Keras.

The pre-trained AlexNet is from:
https://github.com/heuritech/convnets-keras

To run: Add the ILSVRC2012 validation set in a folder named "ILSVRC2012_img_val" (6.3GB), run activations.py (this may take a considerable amount of time) and then run deconvolution.py, to project the maxmial activation of a filter back to image space. 

Deconvolution.py is still a work-in-progress.

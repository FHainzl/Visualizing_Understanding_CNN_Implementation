from keras.models import Model
from keras.layers import Input, Conv2DTranspose
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras import backend as K
from keras.optimizers import SGD

from alexnet import AlexNet

import numpy as np


class DeconvLayers:
    """
    Create deconvolutional layers with parameters extracted from alexnet
    """

    conv_layer_names = AlexNet.conv_layer_names
    deconv_layer_names = AlexNet.deconv_layer_names

    def __init__(self, conv_base_model):

        # Full AlexNet
        self.conv_base_model = conv_base_model
        self.conv_layers = self.init_conv_layers()

        self.weights = self.init_weights_dict()
        self.bias3D = self.init_bias3D_dict()

        self.deconv_layers = self.init_deconv_layers()

    def deconv_layer_model(self, layer_name):
        K.set_image_dim_ordering('th')

        # Get local variables from dictionaries
        conv_layer = self.conv_layers[layer_name]
        w, b = self.weights[layer_name].tuple

        # Get deconvolutional shapes from convolutional shapes
        deconv_from_shape = conv_layer.output_shape[1:]
        deconv_to_shape = conv_layer.input_shape[1:]
        de_layer_name = 'de' + layer_name

        # Build layer
        inputs = Input(shape=deconv_from_shape)
        prediction = Conv2DTranspose(filters=deconv_to_shape[0],  # N_Filters equals N_channels of convolution input
                                     kernel_size=conv_layer.kernel_size,
                                     strides=conv_layer.strides,
                                     activation='relu',
                                     use_bias=False,
                                     name=de_layer_name)(inputs)

        m = Model(input=inputs, output=prediction)

        assert m.get_layer(de_layer_name).get_weights()[0].shape == w.shape
        m.get_layer(de_layer_name).set_weights([w])

        return m

    def init_conv_layers(self):
        conv_layer_dict = {}
        for layer_name in self.conv_layer_names:
            conv_layer_dict[layer_name] = self.conv_base_model.get_layer(layer_name)
        return conv_layer_dict

    def init_weights_dict(self):
        weights_dict = {}
        for layer_name, layer in self.conv_layers.items():
            w = layer.get_weights()[0]
            b = layer.get_weights()[1]
            weights_dict[layer_name] = Weights(w, b)
        return weights_dict

    def init_bias3D_dict(self):
        """
        Arrange bias in 3D shape, such that it can be subtracted from features tensor        
        """

        bias_dict = {}
        for layer_name, layer in self.conv_layers.items():
            b3D = np.zeros(layer.output_shape[1:])
            for f in range(b3D.shape[0]):
                b3D[f] = np.full_like(b3D[0], self.weights[layer_name].b[f])
            b3D = np.expand_dims(b3D, axis=0)
            bias_dict[layer_name] = b3D
        return bias_dict

    def init_deconv_layers(self):
        deconv_layers = {}
        for deconv_layer_name in self.deconv_layer_names:
            deconv_layers[deconv_layer_name] = self.deconv_layer_model(deconv_layer_name[2:])
        return deconv_layers


class Weights:
    def __init__(self, w, b):
        assert type(w) == np.ndarray
        assert type(b) == np.ndarray
        self.w = w
        self.b = b
        self.tuple = (w, b)
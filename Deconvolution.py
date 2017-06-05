from AlexNet import AlexNet, preprocess_image_batch

from keras.models import Model
from keras.layers import Input, Deconv2D
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras import backend as K
from keras.optimizers import SGD

from PIL import Image
import numpy as np
import os


def Deconv_Net(conv_base_model):
    K.set_image_dim_ordering('th')

    conv1 = conv_base_model.get_layer('conv_1')
    # max1 = conv_model.get_layer('m')

    inputs = Input(shape=conv1.output_shape[1:])
    # prediction = Deconv2D(filters=conv1.input_shape[1], kernel_size=conv1.kernel_size, strides=conv1.strides,
    #                       activation='relu',
    #                       name='deconv_1')(inputs)

    # Try without Bias
    prediction = Deconv2D(filters=conv1.input_shape[1], kernel_size=conv1.kernel_size, strides=conv1.strides,
                          activation='relu', use_bias=False,
                          name='deconv_1')(inputs)

    m = Model(input=inputs, output=prediction)
    convert_all_kernels_in_model(m)

    # w_de, b_de = m.get_layer('deconv_1').get_weights()
    w_de = m.get_layer('deconv_1').get_weights()
    w, b = conv_base_model.get_layer('conv_1').get_weights()
    m.get_layer('deconv_1').set_weights([w])

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    m.compile(optimizer=sgd, loss='mse')

    # That was stupid
    # w, b = m.get_weights()


    return m


def get_weights_dict(model):
    weights_dict = {}
    for layer in model.layers:
        if not layer.name.startswith('conv_'):
            print('Not important: {}'.format(layer.name))
            continue
        weight = layer.get_weights()
        try:
            w = weight[0]
            b = weight[1]
            assert type(w) == np.ndarray
            weights_dict[layer.name] = [w, b]
        except IndexError:
            print('The problem was {}'.format(layer.name))
    return weights_dict


"""
Transform output of Deconvolutional Net to image and save to file
"""


def array2image(array, filename='test.JPEG'):
    result = result[0, :, :, :]
    result = np.moveaxis(result, 0, -1)
    result[:, :, 0] += 123.68
    result[:, :, 1] += 116.779
    result[:, :, 2] += 103.939
    print(result.shape)
    filename = 'test.jpeg'
    new_im = Image.fromarray(result.astype(dtype=np.uint8), 'RGB')


class Deconv_Output():
    def __init__(self, output):  # Takes output of DeconvNet
        self.array = self._rearrange_array(output)
        self.image = None

    def _rearrange_array(self, unarranged_array):
        assert len(unarranged_array.shape) in (3, 4)

        # If Array is not yet rearranged
        if len(unarranged_array.shape) == 4:
            assert unarranged_array.shape[0] == 1
            unarranged_array = unarranged_array[0, :, :, :]  # Eliminate batch size dimension
            unarranged_array = np.moveaxis(unarranged_array, 0, -1)  # Put channels last
            # Undo sample mean subtraction
            unarranged_array[:, :, 0] += 123.68
            unarranged_array[:, :, 1] += 116.779
            unarranged_array[:, :, 2] += 103.939

        return unarranged_array

    def save_as(self, folder=None, filename='test.JPEG'):
        self.image = Image.fromarray(self.array.astype(np.uint8, 'RGB'))
        if self.image.mode != 'RGB':
            self.image = self.image.convert('RGB')

        assert type(folder) is str
        if folder is not None:
            filename = folder + '/' + filename

        try:
            os.remove(filename)
        except OSError:
            pass

        self.image.save(filename)


if __name__ == '__main__':
    conv_base_model = AlexNet()
    conv_model = Model(inputs=conv_base_model.input, outputs=conv_base_model.get_layer('conv_1').output)
    deconv_model = Deconv_Net(conv_base_model)
    # deconv_model.summary()
    im = preprocess_image_batch(['Layer1_Strongest_IMG/Layer1_Filter{}_Top7.JPEG'.format(filter)])
    activation = conv_model.predict(im)
    # filter = 1
    # activation_of_one_filter = np.zeros_like(activation)
    # activation_of_one_filter[:, filter - 1, :, :] = activation[:, filter - 1, :, :]
    # # activation_of_one_filter = activation
    # max_activation_of_one_filter = np.zeros_like(activation)
    # max_loc = activation_of_one_filter.argmax()
    # max_loc = np.unravel_index(max_loc,activation.shape)
    # max_activation_of_one_filter[0+max_loc] = activation_of_one_filter.max()
    result = deconv_model.predict(activation)

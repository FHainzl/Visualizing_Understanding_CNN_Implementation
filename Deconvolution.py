from AlexNet import AlexNet, AlexNetModel, preprocess_image_batch
from Deconv_AdditionalLayers import subtract_bias

from keras.models import Model
from keras.layers import Input, Deconv2D
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras import backend as K
from keras.optimizers import SGD

from PIL import Image
import numpy as np
import os


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


class Deconvolution():
    def __init__(self, conv_base_model):
        self.layername = 'conv_1'
        self.conv_layer = conv_base_model.get_layer(self.layername)
        self.conv_from_shape = self.conv_layer.output_shape[1:]
        self.conv_to_shape = self.conv_layer.input_shape[1:]
        self.w, self.b = self.conv_layer.get_weights()
        self.bias3D = self.Bias3D()

        self.conv_base_model = conv_base_model
        self.deconv_model = self.Deconv1_Model()

    def Deconv1_Model(self):
        K.set_image_dim_ordering('th')
        inputs = Input(shape=self.conv_from_shape)
        prediction = Deconv2D(filters=self.conv_to_shape[0],
                              kernel_size=self.conv_layer.kernel_size,
                              strides=self.conv_layer.strides,
                              activation='relu',
                              use_bias=False,
                              name='deconv_1')(inputs)

        m = Model(input=inputs, output=prediction)
        convert_all_kernels_in_model(m)

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        m.compile(optimizer=sgd, loss='mse')

        assert m.get_layer('deconv_1').get_weights()[0].shape == self.w.shape
        w_t = np.moveaxis(self.w,1,0)
        m.get_layer('deconv_1').set_weights([w_t])

        return m

    def Bias3D(self):
        b3D = np.zeros(self.conv_from_shape)
        for f in range(self.b.size):
            b3D[f] = np.full_like(b3D[0], self.b[f])
        b3D = np.expand_dims(b3D, axis=0)
        return b3D

    def predict(self, input):
        starttensor = input - self.bias3D
        bias = self.deconv_model.predict(starttensor)
        return bias


class Deconv_Output():
    def __init__(self, unarranged_array):  # Takes output of DeconvNet
        self.array = self._rearrange_array(unarranged_array)
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
        self.image = Image.fromarray(self.array.astype(np.uint8), 'RGB')
        if self.image.mode != 'RGB':
            self.image = self.image.convert('RGB')

        if folder is not None:
            assert type(folder) is str
            filename = folder + '/' + filename

        try:
            os.remove(filename)
        except OSError:
            pass

        self.image.save(filename)


def visualize_all_filter_in_layer1(conv_model):
    w = conv_model.get_layer('conv_1').get_weights()[0]
    for f in range(96):
        wf = w[:, :, :, f - 1]
        #scale = min(abs(100/wf.max()),abs(100/wf.min()))
        scale = 1000
        wf *= scale
        wf[:, :, 0] += 123.68
        wf[:, :, 1] += 116.779
        wf[:, :, 2] += 103.939
        result = Deconv_Output(wf)
        result.save_as(filename='Filters_Layer1_Visualized/filter{}.JPEG'.format(f+1))


if __name__ == '__main__':
    layer_num = 1
    # activations = AlexNet(layer_num).predict('Example_JPG/Elephant.jpg')

    if False:
        # Visualize first layer filters
        conv_model = AlexNetModel()
        visualize_all_filter_in_layer1(conv_model)

        # Show that inverting preprocessing works
        img_path = 'Example_JPG/Elephant.jpg'
        array = preprocess_image_batch([img_path])
        Deconv_Output(array).save_as(filename='array.JPEG')


    # activation_of_one_filter = np.zeros_like(activations)
    # activation_of_one_filter[:, f - 1, :, :] = activations[:, f - 1, :, :]
    # deconv = Deconvolution(conv_base_model)
    # deconv.project_back(activations)
    # result = Deconv_Output(deconv.project_back(activations))
    # result.save_as()


    # conv_base_model = AlexNet()
    # conv_model = Model(inputs=conv_base_model.input, outputs=conv_base_model.get_layer('conv_1').output)
    # deconv_model = Deconv_Net(conv_base_model)
    # # deconv_model.summary()
    # im = preprocess_image_batch(['Layer1_Strongest_IMG/Layer1_Filter{}_Top7.JPEG'.format(filter)])
    # activation = conv_model.predict(im)
    # # filter = 1
    # # activation_of_one_filter = np.zeros_like(activation)
    # # activation_of_one_filter[:, filter - 1, :, :] = activation[:, filter - 1, :, :]
    # # # activation_of_one_filter = activation
    # # max_activation_of_one_filter = np.zeros_like(activation)
    # # max_loc = activation_of_one_filter.argmax()
    # # max_loc = np.unravel_index(max_loc,activation.shape)
    # # max_activation_of_one_filter[0+max_loc] = activation_of_one_filter.max()
    # result = deconv_model.predict(activation)

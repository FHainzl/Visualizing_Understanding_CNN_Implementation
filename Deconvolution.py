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


class Deconvolution():
    conv_layer_names = ['conv_' + id for id in ('1', '2_1', '2_2', '3', '4_1', '4_2', '5_1', '5_2')]

    def __init__(self, conv_base_model=None):

        # Full AlexNet
        self.conv_base_model = conv_base_model if conv_base_model else AlexNetModel()
        self.conv_layers = self.init_conv_layers()
        # Create models that extract features for each layer
        self.conv_sub_models = [AlexNet(i, self.conv_base_model) for i in (1, 2, 3, 4, 5)]

        self.weights = self.init_weights_dict()
        self.bias3D = self.init_bias3D_dict()
        pass

    def Deconv_Model(self, layer_name):
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
        prediction = Deconv2D(filters=deconv_to_shape[0],  # N_Filters equals N_channels of convolution input
                              kernel_size=conv_layer.kernel_size,
                              strides=conv_layer.strides,
                              activation='relu',
                              use_bias=False,
                              name=de_layer_name)(inputs)

        m = Model(input=inputs, output=prediction)
        convert_all_kernels_in_model(m)

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        m.compile(optimizer=sgd, loss='mse')

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
        for layer_name,layer in self.conv_layers.items():
            w = layer.get_weights()[0]
            b = layer.get_weights()[1]
            weights_dict[layer_name] = W(w, b)
        return weights_dict

    def init_bias3D_dict(self):
        bias_dict = {}
        for layer_name,layer in self.conv_layers.items():
            b3D = np.zeros(layer.output_shape[1:])
            for f in range(b3D.shape[0]):
                b3D[f] = np.full_like(b3D[0], self.weights[layer_name].b[f])
            b3D = np.expand_dims(b3D, axis=0)
            bias_dict[layer_name] = b3D
        return bias_dict

    def predict(self, input):
        starttensor = input - self.bias3D
        bias = self.deconv_model.predict(starttensor)
        return bias


class W():
    def __init__(self, w, b):
        assert type(w) == np.ndarray
        assert type(b) == np.ndarray
        self.w = w
        self.b = b
        self.tuple = (w,b)


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
        # scale = min(abs(100/wf.max()),abs(100/wf.min()))
        scale = 1000
        wf *= scale
        wf[:, :, 0] += 123.68
        wf[:, :, 1] += 116.779
        wf[:, :, 2] += 103.939
        result = Deconv_Output(wf)
        result.save_as(filename='Filters_Layer1_Visualized/filter{}.JPEG'.format(f + 1))


if __name__ == '__main__':
    deconv = Deconvolution()
    layer_num = 1
    activations = AlexNet(layer_num).predict('Example_JPG/Elephant.jpg')
    Deconv_Output(deconv.Deconv_Model('conv_1').predict(activations)).save_as()

    # Filter visualization and image output test
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

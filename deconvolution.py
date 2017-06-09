from alexnet import AlexNet, alexnet_model, preprocess_image_batch

from deconvolution_additional_layers import DeconvLayers

from PIL import Image
import numpy as np
import os


class Deconvolution:
    channels = [3, 96, 256, 384, 384, 256]  # Corresponds to the number of filters in convolution

    def __init__(self, conv_base_model=None):
        # Set convolutional model and submodels, which get activations after given layer
        self.conv_base_model = conv_base_model if conv_base_model else alexnet_model()
        self.conv_sub_models = [None] + [AlexNet(i, self.conv_base_model) for i in (1, 2, 3, 4, 5)]  # Make it 1-based

        # Get deconvolutional layers from Deconv_Layers instance
        DeconvLayers_Instance = DeconvLayers(self.conv_base_model)
        self.deconv_layers = DeconvLayers_Instance.deconv_layers
        self.bias3D = DeconvLayers_Instance.bias3D

        # This attributes will be filled by 'project_down' method
        self.array = None  # Tensor being projected down from feature space to image space
        self.activation_maxpool = None  # Activation for max_pool layer 1 and 2, needed for switches
        self.current_layer = None  # Changes as array is passed on
        self.f = None  # Filter whose activation is projected down

    def project_down(self, image_path, layer, f=None):
        assert type(layer) == int
        self.current_layer = layer
        self.f = f
        self.array = self.conv_sub_models[self.current_layer].predict(image_path)
        self.activation_maxpool = [None] + [self.conv_sub_models[i].predict(image_path) for i in (1, 2)]

        if f:
            self.set_zero_except_maximum()

        if self.current_layer >= 5:
            self.project_through_split_convolution()
            # Zerounpadding
            self.array = self.array[:, :, 1:-1, 1:-1]
        if self.current_layer >= 4:
            self.project_through_split_convolution()
            # Zerounpadding
            self.array = self.array[:, :, 1:-1, 1:-1]
        if self.current_layer >= 3:
            self.project_through_convolution()
            # Zerounpadding
            self.array = self.array[:, :, 1:-1, 1:-1]
        if self.current_layer >= 2:
            self.unpool()
            self.project_through_split_convolution()
            # Zerounpadding
            self.array = self.array[:, :, 2:-2, 2:-2]
        if self.current_layer >= 1:
            self.unpool()
            self.project_through_convolution()
        return self.array

    def zero_un_padd(self):
        # TODO: Implement as method
        pass

    def project_through_convolution(self):
        cl = self.current_layer
        assert cl in (1, 3)

        deconv_cl = 'deconv_{}'.format(cl)
        self.array = self.deconv_layers[deconv_cl].predict(self.array)
        self.current_layer -= 1

    def project_through_split_convolution(self):
        cl = self.current_layer
        assert cl in (2, 4, 5)

        # Make sure dimensions are fine
        assert self.array.shape[1] == self.channels[cl], 'Channel number incorrect'

        # Split, perform Deconvolution on splits, merge
        activation_cl_1 = self.array[:, : self.channels[cl] // 2]
        activation_cl_2 = self.array[:, self.channels[cl] // 2:]

        deconv_cl_1 = 'deconv_{}_1'.format(cl)
        deconv_cl_2 = 'deconv_{}_2'.format(cl)
        projected_activation_cl_1 = self.deconv_layers[deconv_cl_1].predict(activation_cl_1)
        projected_activation_cl_2 = self.deconv_layers[deconv_cl_2].predict(activation_cl_2)

        self.array = np.concatenate((projected_activation_cl_1, projected_activation_cl_2), axis=1)
        assert self.array.shape[1] == self.channels[cl - 1], 'Channel number incorrect'
        self.current_layer -= 1

    def unpool(self):
        cl = self.current_layer
        assert cl in (1, 2), 'Maxpooling only for layer one and two'
        activations = self.activation_maxpool[cl]

        # Network parameters for maxpool layers
        kernel = 3
        stride = 2

        # TODO: Simplify to last 2 lines
        # Change last to lines to assignment once everything works nicely
        assert cl in (1, 2)
        if cl == 1:
            input_shape = (96, 55, 55)
            output_shape = (96, 27, 27)
        if cl == 2:
            input_shape = (256, 27, 27)
            output_shape = (256, 13, 13)
        assert activations.shape[1:] == input_shape
        assert self.array.shape[1:] == output_shape

        reconstructed_activations = np.zeros_like(activations)
        for f in range(output_shape[0]):
            for i_out in range(output_shape[1]):
                for j_out in range(output_shape[2]):
                    i_in, j_in = i_out * stride, j_out * stride
                    sub_square = activations[0, f, i_in:i_in + kernel, j_in:j_in + kernel]
                    max_pos_i, max_pos_j = np.unravel_index(np.nanargmax(sub_square), (kernel, kernel))
                    array_pixel = self.array[0, f, i_out, j_out]

                    # Since poolings are overlapping, two activations might be reconstructed to same spot
                    # Keep the higher activation
                    if reconstructed_activations[0, f, i_in + max_pos_i, j_in + max_pos_j] < array_pixel:
                        reconstructed_activations[0, f, i_in + max_pos_i, j_in + max_pos_j] = array_pixel
        self.array = reconstructed_activations

    def set_zero_except_maximum(self):
        # Set other layers to zero
        new_array = np.zeros_like(self.array)
        new_array[0, self.f - 1] = self.array[0, self.f - 1]

        #Set other activations in same layer to zero
        max_index_flat = np.nanargmax(new_array)
        max_index = np.unravel_index(max_index_flat, new_array.shape)
        self.array = np.zeros_like(new_array)
        self.array[max_index] = new_array[max_index]


class DeconvOutput:
    def __init__(self, unarranged_array, contrast=1):  # Takes output of DeconvNet
        self.contrast = contrast
        self.array = self._rearrange_array(unarranged_array)
        self.image = None

    def _rearrange_array(self, unarranged_array):
        assert len(unarranged_array.shape) in (3, 4)

        # If Array is not yet rearranged
        if len(unarranged_array.shape) == 4:
            assert unarranged_array.shape[0] == 1
            unarranged_array = unarranged_array[0, :, :, :]  # Eliminate batch size dimension
            unarranged_array = np.moveaxis(unarranged_array, 0, -1)  # Put channels last

            unarranged_array *= self.contrast
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


def visualize_all_filters_in_layer1(conv_model):
    w = conv_model.get_layer('conv_1').get_weights()[0]
    for f in range(96):
        wf = w[:, :, :, f - 1]
        # scale = min(abs(100/wf.max()),abs(100/wf.min()))
        scale = 1000
        wf *= scale
        wf[:, :, 0] += 123.68
        wf[:, :, 1] += 116.779
        wf[:, :, 2] += 103.939
        result = DeconvOutput(wf)
        result.save_as(filename='Filters_Layer1_Visualized/filter{}.JPEG'.format(f + 1))


if __name__ == '__main__':
    layer = 5
    f = 155
    img_path = 'Layer{}_Strongest_max_IMG/Layer{}_Filter{}_Top1.JPEG'.format(layer, layer, f)
    array = Deconvolution().project_down(img_path, layer, f)
    DeconvOutput(array, contrast=5).save_as()

    # Filter visualization and image output test
    if False:
        # Visualize first layer filters
        conv_model = alexnet_model()
        visualize_all_filters_in_layer1(conv_model)

        # Show that inverting preprocessing works
        img_path = 'Example_JPG/Elephant.jpg'
        array = preprocess_image_batch([img_path])
        DeconvOutput(array).save_as(filename='array.JPEG')

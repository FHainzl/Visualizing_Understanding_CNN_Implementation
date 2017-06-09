"""
Code based on 
https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/convnets.py
"""

from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, merge
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D

from keras.optimizers import SGD
from keras import backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils import plot_model

import numpy as np
from scipy.misc import imread, imresize

from alexnet_additional_layers import split_tensor, cross_channel_normalization
from decode_predictions import decode_classnames_json


def alexnet_model(weights_path=None):
    """
        Returns a keras model for AlexNet.
    """
    K.set_image_dim_ordering('th')
    inputs = Input(shape=(3, 227, 227))

    conv_1 = Conv2D(96, 11, strides=4, activation='relu', name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = cross_channel_normalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = merge([
        Conv2D(128, 5, activation="relu", name='conv_2_' + str(i + 1))
            (split_tensor(ratio_split=2, id_split=i)(conv_2)
        ) for i in range(2)], mode='concat', concat_axis=1, name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = cross_channel_normalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, 3, activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = merge([
        Conv2D(192, 3, activation="relu", name='conv_4_' + str(i + 1))(
            split_tensor(ratio_split=2, id_split=i)(conv_4)
        ) for i in range(2)], mode='concat', concat_axis=1, name="conv_4")

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = merge([
        Conv2D(128, 3, activation="relu", name='conv_5_' + str(i + 1))(
            split_tensor(ratio_split=2, id_split=i)(conv_5)
        ) for i in range(2)], mode='concat', concat_axis=1, name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000, name='dense_3')(dense_3)
    prediction = Activation("softmax", name="softmax")(dense_3)

    m = Model(input=inputs, output=prediction)

    if weights_path is None:
        weights_path = 'Data/alexnet_weights.h5'
    m.load_weights(weights_path)
    # Model was trained using Theano backend
    # This changes convolutional kernels from TF to TH, great accuracy improvement
    convert_all_kernels_in_model(m)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    m.compile(optimizer=sgd, loss='mse')

    return m


def preprocess_image_batch(image_paths, img_size=(256, 256), crop_size=(227, 227), color_mode="rgb", out=None):
    img_list = []

    for im_path in image_paths:
        img = imread(im_path, mode='RGB')
        img = imresize(img, img_size)

        img = img.astype('float32')
        # We normalize the colors (in RGB space) with the empirical means on the training set
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        # We permute the colors to get them in the BGR order
        if color_mode == "bgr":
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
        img = img.transpose((2, 0, 1))

        if crop_size:
            img = img[:, (img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2
            , (img_size[1] - crop_size[1]) // 2:(img_size[1] + crop_size[1]) // 2]

        img_list.append(img)

    try:
        img_batch = np.stack(img_list, axis=0)
    except:
        raise ValueError('when img_size and crop_size are None, images'
                         ' in image_paths must have the same shapes.')

    if out is not None and hasattr(out, 'append'):
        out.append(img_batch)
    else:
        return img_batch


class AlexNet():
    def __init__(self, highest_layer_num=None, base_model = None):
        self.highest_layer_num = highest_layer_num
        self.base_model = base_model if base_model else alexnet_model() # If no base model specified, use AlexNet
        self.model = self.sub_model() if highest_layer_num else self.base_model # Use full network if no highest_layer

    def sub_model(self):
        highest_layer_name = 'conv_{}'.format(self.highest_layer_num)
        highest_layer = self.base_model.get_layer(highest_layer_name)
        return Model(inputs=self.base_model.input,
                     outputs=highest_layer.output)

    def predict(self, img_path):
        img = preprocess_image_batch([img_path])
        return self.model.predict(img)



if __name__ == "__main__":
    # Test pre-trained model
    im = preprocess_image_batch(['Example_JPG/Elephant.jpg'])
    model = alexnet_model()
    # plot_model(model, to_file='model.png',show_shapes=True)
    # print(model.summary())
    out = model.predict(im)
    print(decode_classnames_json(out))

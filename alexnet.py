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
from decode_predictions import decode_classnames_json, decode_classnumber


def alexnet_model(weights_path=None):
    """
    Returns a keras model for AlexNet, achieving roughly 80% at ImageNet2012 validation set
    
    Model and weights from
    https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/convnets.py
    and only slightly modified to work with TF backend
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

    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #m.compile(optimizer=sgd, loss='mse')

    return m


def preprocess_image_batch(image_paths, img_size=(256, 256), crop_size=(227, 227), color_mode="rgb", out=None):
    """
    Resize, crop and normalize colors of images 
    to make them suitable for alexnet_model (if default parameter values are chosen)
    
    This function is also from 
    https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/convnets.py
    with only some minor changes
    """

    # Make function callable with single image instead of list
    if type(image_paths) is str:
        image_paths = [image_paths]

    img_list = []
    for im_path in image_paths:
        img = imread(im_path, mode='RGB')
        img = imresize(img, img_size)

        img = img.astype('float32')
        # Normalize the colors (in RGB space) with the empirical means on the training set
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
    """
    Wrapper for alexnet_model, makes calculating features of intermediate layers a one-liner
    Call with alexnet_model, if one already exists; otherwise one will be created
    If highest layer is given, predictions() will return output of convolution at that layer
    If not, predictions() will return 1000-dim vector of hot-encoded class probabilities
    """

    val_set_size = 50000
    filter_per_layer = {1: 96, 2: 256, 3: 384, 4: 384, 5: 256}
    conv_layer_names = ['conv_' + id for id in ('1', '2_1', '2_2', '3', '4_1', '4_2', '5_1', '5_2')]
    deconv_layer_names = ['deconv_' + id for id in ('1', '2_1', '2_2', '3', '4_1', '4_2', '5_1', '5_2')]

    def __init__(self, highest_layer_num=None, base_model=None):
        self.highest_layer_num = highest_layer_num
        self.base_model = base_model if base_model else alexnet_model()  # If no base_model, create alexnet_model
        self.model = self._sub_model() if highest_layer_num else self.base_model  # Use full network if no highest_layer

    def _sub_model(self):
        highest_layer_name = 'conv_{}'.format(self.highest_layer_num)
        highest_layer = self.base_model.get_layer(highest_layer_name)
        return Model(inputs=self.base_model.input,
                     outputs=highest_layer.output)

    def predict(self, img_path):
        """
        Takes the image path as argument, unlike alexnet_model.predict which takes the preprocessed array
        """
        img = preprocess_image_batch(img_path)
        return self.model.predict(img)

    def top_classes(self,img_path,top=5):
        preds = self.predict(img_path)
        return decode_classnumber(preds,top)


if __name__ == "__main__":
    img_path = 'Example_JPG/Elephant.jpg'

    # Usage of alexnet_model
    im = preprocess_image_batch([img_path])
    model = alexnet_model()
    out_model = model.predict(im)

    # Usage of AlexNet()
    out_wrapper = AlexNet().predict(img_path)

    assert (out_model == out_wrapper).all()

    # Decode one-hot vector to most probable class names
    print(decode_classnames_json(out_wrapper))
    print(decode_classnumber(out_wrapper))

    # Plot and print information about model
    plot_and_print = False
    if plot_and_print:
        plot_model(model, to_file='model.png', show_shapes=True)
        print(model.summary())

    testimages = ['Example_JPG/Elephant.jpg', 'Example_JPG/RoadBike.jpg', 'Example_JPG/Trump.jpg']
    model = alexnet_model()
    preds = AlexNet(base_model=model).top_classes(testimages)
    print(preds)
    for pred in preds:
        print(pred)

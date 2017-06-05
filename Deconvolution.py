from AlexNet import AlexNet, preprocess_image_batch

from keras.models import Model
from keras.layers import Input, Deconv2D
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras import backend as K
from keras.optimizers import SGD

from PIL import Image
import numpy as np
import os

def Deconv_Net(conv_model):
    K.set_image_dim_ordering('th')

    conv1 = conv_model.get_layer('conv_1')
    max1 = conv_model.get_layer('m')
    inputs = Input(shape=conv1.output_shape[1:])
    prediction = Deconv2D(filters=conv1.input_shape[1], kernel_size=conv1.kernel_size, strides=conv1.strides,
                          activation='relu',
                          name='deconv_1')(inputs)

    m = Model(input=inputs, output=prediction)
    convert_all_kernels_in_model(m)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    m.compile(optimizer=sgd, loss='mse')

    w, b = m.get_weights()
    m.set_weights([w, b])

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


if __name__ == '__main__':
    conv_base_model = AlexNet()
    conv_model = Model(inputs=conv_base_model.input, outputs=conv_base_model.get_layer('conv_1').output)
    deconv_model = DeConv_1(conv_base_model)
    # deconv_model.summary()

    im = preprocess_image_batch(['Layer1_Strongest_IMG/Layer1_Filter43_Top1.JPEG'])
    activation = conv_model.predict(im)
    #activation_of_one_filter = np.zeros_like(activation)
    #activation_of_one_filter[:,42,:,:] = activation[:,42,:,:]
    activation_of_one_filter = activation
    result = deconv_model.predict(activation_of_one_filter)
    print(result.shape)
    result = result[0, :, :, :]
    result = np.moveaxis(result, 0, -1)
    result[:, :, 0] += 123.68
    result[:, :, 1] += 116.779
    result[:, :, 2] += 103.939
    print(result.shape)
    filename = 'test.jpeg'
    new_im = Image.fromarray(result.astype(dtype=np.uint8), 'RGB')
    # for i in range(3):
    #     new_im = Image.fromarray(result[:,:,i])
    if new_im.mode != 'RGB':
        new_im = new_im.convert('RGB')
    #     new_im.save("test{}.jpeg".format(i))
    try:
        os.remove(filename)
    except OSError:
        pass
    new_im.save(filename)
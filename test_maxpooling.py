from alexnet import alexnet_model
from keras.models import Model
import numpy as np
from time import time


class test_maxpooling():
    def __init__(self):
        helper = np.random.random((1, 96, 27, 27))
        self.array = helper.copy()

    def max_pool(self):
        # Network parameters for maxpool layers
        kernel = 3
        stride = 2

        # For testing
        layer = 1

        helper2 = np.random.random((1, 96, 55, 55))
        activations = helper2.copy()
        self.activations = activations

        assert layer in (1, 2)
        if layer == 1:
            input_shape = (96, 55, 55)
            output_shape = (96, 27, 27)
        if layer == 2:
            input_shape = (256, 27, 27)
            output_shape = (256, 13, 13)
        assert activations.shape[1:] == input_shape
        assert self.array.shape[1:] == output_shape

        reconstructed_activations = np.zeros_like(activations)
        # for f in range(output_shape[0]):
        f = 0
        if f == 0:
            for i_out in range(output_shape[1]):
                for j_out in range(output_shape[2]):
                    i_in, j_in = i_out * stride, j_out * stride

                    print("i,j in: ", i_in, j_in, "i,j out: ", i_out, j_out)

                    sub_square = activations[0, f, i_in:i_in + kernel, j_in:j_in + kernel]
                    max_pos_i, max_pos_j = np.unravel_index(np.nanargmax(sub_square), (kernel, kernel))

                    print('Max ', max_pos_i, max_pos_j)
                    array_pixel = self.array[0, f, i_out, j_out]

                    # Since poolings are overlapping, two activations might be reconstructed to same spot
                    # Keep the higher activation
                    if reconstructed_activations[0, f, i_in + max_pos_i, j_in + max_pos_j] < array_pixel:
                        reconstructed_activations[0, f, i_in + max_pos_i, j_in + max_pos_j] = array_pixel
        self.reconstructed_activations = reconstructed_activations


if __name__ == '__main__':
    # base_model = AlexNetModel()
    # max_pool_1 = base_model.get_layer('max_pooling2d_1')
    # model = Model(inputs=max_pool_1.input, outputs=max_pool_1.output)
    # inp = np.random.random((1, 96, 55, 55))
    # outp = model.predict(inp)
    x = test_maxpooling()
    start = time()
    x.max_pool()
    print(time() - start)

from keras.layers.core import Lambda
from keras import backend as K


#
# def multiply(x, n):
#     x_prime = tf.reshape(x, (-1, n, 1))
#     x_transpose = tf.transpose(x_prime, perm=[0, 2, 1])
#     return tf.batch_matmul(x_transpose, x_prime)
#
#
# Lambda(lambda x: multiply(x, n), output_shape=(n, n))


def subtract_bias(bias_tensor, **kwargs):
    K.set_image_dim_ordering('th')
    def output_shape(input_shape):
        K.set_image_dim_ordering('th')
        print(input_shape, type(input_shape))
        shape = list(input_shape)
        return tuple(shape)

    return Lambda(
        lambda x: K.update_sub(x, bias_tensor))  # ,output_shape=lambda input_shape: output_shape(input_shape))
    #
    # def f(X):
    #     div = int(X.get_shape()[axis]) // ratio_split
    #
    #     if axis == 0:
    #         output = X[id_split * div:(id_split + 1) * div, :, :, :]
    #     elif axis == 1:
    #         output = X[:, id_split * div:(id_split + 1) * div, :, :]
    #     elif axis == 2:
    #         output = X[:, :, id_split * div:(id_split + 1) * div, :]
    #     elif axis == 3:
    #         output = X[:, :, :, id_split * div:(id_split + 1) * div]
    #     else:
    #         raise ValueError("This axis is not possible")
    #
    #     return output

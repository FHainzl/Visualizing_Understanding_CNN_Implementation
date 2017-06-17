from alexnet import AlexNet
from validation import get_labels, get_path_from_id
from activations import get_strongest_filter
from decode_predictions import decode_classnames_json, decode_classnumber
import matplotlib.pyplot as plt
from deconvolution import Deconvolution, DeconvOutput

import numpy as np
from scipy.misc import imread, imresize


def preprocess_image_batch_grey_square(image_paths, square_x, square_y, img_size=(256, 256), crop_size=(227,
                                                                                                        227),
                                       color_mode="rgb", out=None

                                       ):  # Make function callable with single image instead of list


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

        # Add grey square
        assert 12 < square_x < 243
        assert 12 < square_y < 243
        img[:, square_x - 13:square_x + 14, square_y - 13:square_y + 14] = 0

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


def get_summed_activation_of_feature_map(top_layer_model, f, preprocd_img):
    # Get activations for shortened model
    activation_img = top_layer_model.predict(preprocd_img)
    activation_img = activation_img[0, f - 1, :, :]
    activation_img = activation_img.sum(1)
    activation_img = activation_img.sum(0)
    return activation_img


def visualize_one_image(conv_base, path, layer, f, contrast, output_name):
    save_to_folder = 'Occlusion/'

    projection = Deconvolution(conv_base).project_down(path, layer, f)

    activation_filename = save_to_folder + output_name
    DeconvOutput(projection, contrast).save_as(filename=activation_filename + '.JPEG')


def get_heatmaps(img_id, alexnet, title):
    base_model = alexnet.base_model
    top_layer_model = alexnet.model
    labels = get_labels()

    path = get_path_from_id(img_id)

    strongest_filter = get_strongest_filter(img_id, layer=5)
    true_label = labels[img_id]
    print(strongest_filter, true_label)

    predictions = AlexNet(base_model=base_model).predict(path)
    print(decode_classnames_json(predictions))
    print(decode_classnumber(predictions))
    print(true_label)

    # DeconvOutput(preprocess_image_batch_grey_square(image_paths=path, square_x=50, square_y=50)).save_as('Occlusion',
    # title + '.JPEG')
    activations = np.zeros((30, 30))
    class_prop = np.zeros((30, 30))

    for x in range(0, 30):
        print(x)
        for y in range(0, 30):
            prep_image = preprocess_image_batch_grey_square(path, 13 + x * 7, 13 + y * 7)
            activation = get_summed_activation_of_feature_map(top_layer_model, strongest_filter, prep_image)
            prediction = base_model.predict(prep_image)
            activations[x, y] = activation
            class_prop[x, y] = prediction[0][true_label]
    print('done')

    fig, ax = plt.subplots()
    cax = ax.imshow(activations, interpolation='nearest', cmap='plasma')
    plt.axis('off')
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax)
    plt.show()

    fig, ax = plt.subplots()
    cax = ax.imshow(class_prop, interpolation='nearest', cmap='plasma')
    plt.axis('off')
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax)
    plt.show()


if __name__ == '__main__':
    img_id = 9997
    alexnet = AlexNet(5)
    base_model = alexnet.base_model
    top_layer_model = alexnet.model

    get_heatmaps(img_id, alexnet, 'Tractor')

    # path = get_path_from_id(img_id)
    # visualize_one_image(base_model, path, 5, get_strongest_feature_map(img_id), contrast=22, output_name='Berg_Deconv_1')

from alexnet import alexnet_model, preprocess_image_batch

from keras.models import Model
from shutil import copyfile, rmtree

import numpy as np
import os
import time
import pandas


def get_activations(layer_num, base_model, folder='ILSVRC2012_img_val'):
    print('Working on layer ' + str(layer_num))
    val_set_size = 50000
    filter_per_layer = {1: 96, 2: 256, 3: 384, 4: 384, 5: 256}
    filters = filter_per_layer[layer_num]
    matrix_filename = 'Data/Strongest_Activation_Layer{}.csv'.format(layer_num)

    # Init array to save activations, Filter and image numbers start with 1!!
    activations = np.full((val_set_size + 1, filters + 1), fill_value=0.0, dtype=np.float32)

    # Create Model up to layer_num
    layer_name = 'conv_' + str(layer_num)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

    # For timing
    timesteps = [time.time()]
    save_interval = 10000
    time_interval = 1000
    for img_id in range(1, val_set_size + 1):

        # Convert image_id to file location
        id_str = str(img_id)
        while len(id_str) < 5:
            id_str = '0' + id_str
        img_name = folder + '/ILSVRC2012_val_000' + id_str + '.JPEG'

        # Get activations for shortened model
        x = preprocess_image_batch([img_name])
        activation_img = model.predict(x)

        # First approach, the summed activation of a filter. I have now changed to finding maximum activation
        # Sum over image region to get activation for given filter and given image
        # activation_img = activation_img.sum(3)
        # activation_img = activation_img.sum(2)

        # Find maximum activation for each filter for a given image
        activation_img = np.nanmax(activation_img, axis=3)
        activation_img = np.nanmax(activation_img, axis=2)

        # Remove batch size dimension
        assert activation_img.shape[0] == 1
        activation_img = activation_img.sum(0)

        # Make activations 1-based indexing
        activation_img = np.insert(activation_img, 0, 0.0)

        # Copy activation results for image to matrix
        activations[img_id] = activation_img[:]

        if img_id % time_interval == 0:
            timesteps.append(time.time())

            last_interval = timesteps[-1] - timesteps[-2]
            total_execution = timesteps[-1] - timesteps[0]
            expected_remaining_time = total_execution / img_id * (val_set_size - img_id)

            print("Current image ID: {}".format(img_id))
            print("The last 100 images took {0:.2f} s.".format(100 * last_interval / time_interval))
            print("The average time per 100 images so far is {0:.2f} s".format(100 * total_execution / img_id))
            print("Total execution time so far: {0:.2f} min".format(total_execution / 60))
            print("The extrapolated remaining time is {0:.2f} min".format(expected_remaining_time / 60))
            print('\n')

        if img_id % save_interval == 0:
            print("Results for first {} images saved\n".format(img_id))
            try:
                os.remove(matrix_filename)
            except OSError:
                pass
            np.savetxt(matrix_filename, activations, delimiter=',')

    np.savetxt(matrix_filename, activations, delimiter=',')


def find_strongest_image(layer_num, top=9):
    filter_per_layer = {1: 96, 2: 256, 3: 384, 4: 384, 5: 256}
    filters = filter_per_layer[layer_num]

    matrix_filename = 'Data/Strongest_Activation_Layer{}.csv'.format(layer_num)
    read_from_folder = 'ILSVRC2012_img_val'
    save_to_folder = 'Layer' + str(layer_num) + '_Strongest_IMG'

    img_id = 0
    with open(matrix_filename, mode='r') as data:
        activations = pandas.read_csv(matrix_filename, dtype=np.float32, header=None).as_matrix()

    argsorted_activations = activations.argsort(axis=0)
    top_indices = [indices[-top:][::-1] for indices in argsorted_activations.T]
    top_indices[0] = None

    try:
        rmtree(save_to_folder)
    except:
        pass
    os.makedirs(save_to_folder)

    for f in range(1, filters + 1):
        t = 1
        for img_id in top_indices[f]:
            img_name = str(img_id)
            while len(img_name) < 5:
                img_name = '0' + img_name
            copy_from = read_from_folder + '/ILSVRC2012_val_000' + img_name + '.JPEG'

            copy_to = save_to_folder + '\Layer' + str(layer_num) + '_Filter' + str(f) + '_Top' + str(t) + '.JPEG'
            t += 1
            copyfile(copy_from, copy_to)


if __name__ == '__main__':
    base_model = alexnet_model()
    # Get activations and copy maximally activating images to folder
    for i in (5, 4, 3, 2, 1):
        get_activations(i, base_model)
        start = time.time()
        find_strongest_image(i)
        print("Copied images in {} s".format(time.time() - start))

    # Just making sure...
    # lay = base_model.layers
    # for l in lay:
    #     print(l.name)
    # str1 = 'conv_1'
    # str2 = 'max_pooling2d_1'
    # print(base_model.get_layer(name=str1).output == base_model.get_layer(name=str2).input)
    # str1 = 'conv_2'
    # str2 = 'max_pooling2d_2'
    # print(base_model.get_layer(str1).output == base_model.get_layer(str2).input)
    # str1 = 'conv_3'
    # str2 = 'zero_padding2d_3'
    # print(base_model.get_layer(str1).output == base_model.get_layer(str2).input)
    # str1 = 'conv_4'
    # str2 = 'zero_padding2d_4'
    # print(base_model.get_layer(str1).output == base_model.get_layer(str2).input)
    # str1 = 'conv_5'
    # str2 = 'convpool_5'
    # print(base_model.get_layer(str1).output == base_model.get_layer(str2).input)

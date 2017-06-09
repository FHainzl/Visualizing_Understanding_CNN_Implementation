from alexnet import alexnet_model

from shutil import copyfile, rmtree

import numpy as np
import os
import time
import pandas

from alexnet import AlexNet  # For hyper-parameters


def get_activations(layer_num, base_model, mode='summation', folder='ILSVRC2012_img_val'):
    """
    Returns a matrix s.t. activations[i][j] is the summed/maxed activation of image[i] at filter[j] for a given layer
    This matrix of activations is saved as a .csv file and will later be used by 'find_strongest_images'
    :param layer_num: Specifies layer, whose filters' activations are returned
    :param base_model: Pass alexnet_model 
    :param mode: either 'summation' or 'maximum'
    :param folder: Specify folder that contains validation images
    """

    assert mode in ('summation', 'maximum'), "Mode has to be either 'summation' or 'maximum'"

    print('Working on layer {}\N'.format(layer_num))
    filters = AlexNet.filter_per_layer[layer_num]
    activation_matrix_filename = 'Data/Strongest_Activation_Layer{}.csv'.format(layer_num)

    # Init array to save activations, Filter and image numbers start with 1!!
    activations = np.full((AlexNet.val_set_size + 1, filters + 1), fill_value=0.0, dtype=np.float32)

    # Create Model up to layer_num
    model = AlexNet(layer_num, base_model)

    # For timing
    timesteps = [time.time()]
    save_interval = 10000
    time_interval = 1000
    for img_id in range(1, AlexNet.val_set_size + 1):

        # Convert image_id to file location
        id_str = str(img_id)
        while len(id_str) < 5:
            id_str = '0' + id_str
        img_name = folder + '/ILSVRC2012_val_000' + id_str + '.JPEG'

        # Get activations for shortened model
        activation_img = model.predict(img_name)

        # Make sure that dimensions 2 and 3 are spacial (Image is square)
        assert activation_img.shape[2] == activation_img.shape[3], "Index ordering incorrect"

        if mode == 'summing':
            # Sum over spacial dimension to get activation for given filter and given image
            assert activation_img.shape[2] == activation_img.shape[3], "Index ordering incorrect"
            activation_img = activation_img.sum(3)
            activation_img = activation_img.sum(2)

        if mode == 'maximum':
            # Find maximum activation for each filter for a given image
            activation_img = np.nanmax(activation_img, axis=3)
            activation_img = np.nanmax(activation_img, axis=2)

        # Remove batch size dimension
        assert activation_img.shape[0] == 1
        activation_img = activation_img.sum(0)

        # Make activations 1-based indexing
        activation_img = np.insert(activation_img, 0, 0.0)

        #  activation_image is now a vector of length equal to number of filters (plus one for one-based indexing)
        #  each entry corresponds to the maximum/summed activation of each filter for a given image form validation set

        # Copy activation results for image to matrix
        activations[img_id] = activation_img[:]

        # Print progress
        if img_id % time_interval == 0:
            timesteps.append(time.time())

            last_interval = timesteps[-1] - timesteps[-2]
            total_execution = timesteps[-1] - timesteps[0]
            expected_remaining_time = total_execution / img_id * (AlexNet.val_set_size - img_id)

            print("Current image ID: {}".format(img_id))
            print("The last 100 images took {0:.2f} s.".format(100 * last_interval / time_interval))
            print("The average time per 100 images so far is {0:.2f} s".format(100 * total_execution / img_id))
            print("Total execution time so far: {0:.2f} min".format(total_execution / 60))
            print("The extrapolated remaining time is {0:.2f} min".format(expected_remaining_time / 60))
            print('\n')

        # Update output file
        if img_id % save_interval == 0:
            print("Results for first {} images saved\n".format(img_id))
            try:
                os.remove(activation_matrix_filename)
            except OSError:
                pass
            np.savetxt(activation_matrix_filename, activations, delimiter=',')

    np.savetxt(activation_matrix_filename, activations, delimiter=',')


def find_strongest_image(layer_num, top=9, folder='ILSVRC2012_img_val'):
    """
    Reads the data from .csv file returned by 'get_activations' and copies strongest activating images in folders
    :param layer_num: 
    :param top: Number of highest images per filter moved to folder
    :param folder: Specify folder that contains validation images
    """

    filters = AlexNet.filter_per_layer[layer_num]

    activation_matrix_filename = 'Data/Strongest_Activation_Layer{}.csv'.format(layer_num)
    read_from_folder = folder
    save_to_folder = 'Layer{}_Strongest_IMG'.format(layer_num)

    with open(activation_matrix_filename, mode='r'):
        activations = pandas.read_csv(activation_matrix_filename, dtype=np.float32, header=None).as_matrix()

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
        print("Copied images in {} s\n".format(time.time() - start))

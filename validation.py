from alexnet import AlexNet
import os
import time





def predict_validation_set(model, batch_size=500, break_after_batches=None, folder='ILSVRC2012_img_val',
                           filename='Data/ILSVRC2012_img_val_predictions.txt'):
    """
    Predict Top5 most likely classes for images in Validation Set and save them to text file
    """
    val_set_size = 50000

    # Delete file if exists:
    try:
        os.remove(filename)
    except OSError:
        pass

    with open(filename, 'a') as results_file:
        start = []
        finish = []
        for batch in range(val_set_size // batch_size):

            start.append(time.time())

            # Early break for testing
            if break_after_batches is not None and batch == break_after_batches:
                break

            img_batch_locations = []

            for i in range(1 + batch * batch_size, 1 + (batch + 1) * batch_size):
                img_id = str(i)
                while len(img_id) < 5:
                    img_id = '0' + img_id
                img_batch_locations.append(folder + '/ILSVRC2012_val_000' + img_id + '.JPEG')

            batch_predictions = AlexNet(model).top_classes(img_batch_locations)

            for i in range(batch_size):
                results_file.write(
                    img_batch_locations[i][len(folder) + 1:] + ' ' + str(batch_predictions[i])[1:-1] + '\n')

            finish.append(time.time())

            last_execution = finish[-1] - start[-1]
            total_execution = finish[-1] - start[0]
            expected_remaining_time = total_execution / (batch + 1) * (val_set_size // batch_size - batch)

            print("The execution of batch #{} took {} s".format(batch + 1, last_execution / 60))
            print("This corresponds to a time of {} s per 100 images".format(100 * last_execution / batch_size))
            print("Total execution time so far: {} min".format(total_execution / 60))
            print("The extrapolated remaining time is {} min".format(expected_remaining_time / 60))
            print('\n')

def get_path_from_id(img_id):
    img_id_str = str(img_id)
    while len(img_id_str) < 5:
        img_id_str = '0' + img_id_str

    folder = 'ILSVRC2012_IMG_VAL/'
    file = 'ILSVRC2012_val_000' + img_id_str + '.JPEG'

    path = folder + file

    return path


def get_labels():
    """
    Imports labels for ILSVRC2012 Validation set and stores them in list
    Label is stored at array index corresponding to each image's ID
    """
    result = [None] * (AlexNet.val_set_size+1)
    with open('Data/ILSVRC2012_img_val_labels.txt', 'r') as data:
        for line in data:
            filename = line.split(' ')[0]
            im_id = int(filename[-11:-5])
            label = int(line.split(' ')[1])
            result[im_id] = label
    return result





def accuracy_validation_set():
    """
    Determine Accuracy of Top5 predictions on Validation set.
    This is NOT an honest test accuracy, because validation can be used in training. Labels of test set are not available.
    """

    labels = get_labels()
    filename = 'Data/ILSVRC2012_img_val_predictions.txt'
    total = 0
    correct = 0
    with open(filename, 'r') as data:
        for line in data:
            filename = line.split(' ')[0]
            im_id = int(filename[-11:-5])
            predictions = [int(num.strip()) for num in line.split()[1:]]
            if labels[im_id] in predictions:
                correct += 1
            total += 1
        return correct / total


if __name__ == '__main__':
    # Generate prediction and save to text file
    if False:
        model = AlexNet().model
        predict_validation_set(model)
    # Print accuracy
    print('Accuracy at Validation Set: {}%'.format(accuracy_validation_set()))

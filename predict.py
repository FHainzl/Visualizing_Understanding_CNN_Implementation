from VGG16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input

from alexnet import alexnet_model, preprocess_image_batch

import numpy as np
from decode_predictions import decode_classnames_json, decode_classnumber

from keras.preprocessing import image as keras_image


def classify_VGG16(img_path, model=None, printing=False):
    if model is None:
        model = VGG16(include_top=True, weights='imagenet')

    img = keras_image.load_img(img_path, target_size=(224, 224))  # Loads an image into PIL format.
    x = keras_image.img_to_array(img)  # Converts a PIL Image instance to a Numpy array.
    x = np.expand_dims(x, axis=0)  # Add dimension for batch size
    x = vgg16_preprocess_input(x)
    if printing:
        print('Input image shape:', x.shape)

    raw_preds = model.predict(x)
    preds = decode_classnumber(raw_preds)

    if printing:
        preds = decode_classnames_json(raw_preds)
        print('VGG16 Predicted:')
        print(preds)

    return preds


def classify_alexnet(img_path, model=None, printing=False):
    if model is None:
        model = alexnet_model()

    # Convert img_path to iterable if necessary
    if type(img_path) is str:
        img_path = [img_path]

    x = preprocess_image_batch(img_path)
    if printing:
        print('Input image shape:', x.shape)

    raw_preds = model.predict(x)
    preds = decode_classnumber(raw_preds)

    if printing:
        preds = decode_classnames_json(raw_preds)
        print('AlexNet Predicted:')
        for pred in preds:
            print(pred)

    return preds


def print_top5(img_path, CNN='AlexNet'):
    assert CNN in ('AlexNet', 'VGG16')
    if CNN == 'AlexNet':
        classify_alexnet(img_path, printing=True)
    else:
        classify_VGG16(img_path, printing=True)



if __name__ == '__main__':
    # testimage = 'Example_JPG/Cabrio.jpg'
    # Print_Top5(testimage, 'AlexNet')
    # Print_Top5(testimage, 'VGG16')

    testimages = ['Example_JPG/Cabrio.jpg', 'Example_JPG/RoadBike.jpg', 'Example_JPG/Trump.jpg']
    model = alexnet_model()
    preds = (classify_alexnet(testimages, model))
    for pred in preds:
        print(pred)
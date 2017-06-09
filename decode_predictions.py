import json

"""
Helper functions to convert 1000dim vector with class probabilities (or batches thereof) to list of top classes

Based on 
decode_predictions from keras.applications.imagenet_utils 
"""


def decode_classnames_json(preds, top=5):
    """
    Returns class code, class name and probability for each class amongst top=5 for each prediction in preds
    
    e.g.
    [[('n01871265', 'tusker', 0.69987053), ('n02504458', 'African_elephant', 0.18252705), ... ]]
    """

    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_classnames_json` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))

    with open('Data/imagenet_class_index.json') as data_file:
        data = json.load(data_file)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(data[str(i)]) + (pred[i],) for i in top_indices]
        results.append(result)
    return results


def decode_classnumber(preds, top=5):
    """
    Return class number between 0 and 999 for each class amongst top=5 for each prediction in preds
    
    e.g.
    [[101, 386, 385, 149, 343]]
    """

    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_classnumber` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = list(top_indices)
        results.append(result)
    return results

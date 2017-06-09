import json

"""
Based on keras.applications.imagenet_utils
"""


def decode_classnames_json(preds, top=5):
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
        result1 = list(result)
        result1.sort(key=lambda x: x[2], reverse=True)  # Isn't this redundant? Should be already sorted
        assert result == result1
        results.append(result1)
    return results


def decode_classnumber(preds, top=5):
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_classnumber` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        results.append(top_indices)
    return results

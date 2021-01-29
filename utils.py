from preprocess import lineToTensor, labelToTensor, letterToIndex


def meta():
    import json
    from collections import OrderedDict
    with open("dataset/meta.json", "r") as File:
        Dict = json.loads(File.read(), object_pairs_hook=OrderedDict)
    return Dict

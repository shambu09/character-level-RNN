import time
import math
import json
import torch
import random
from preprocess import lineToTensor, letterToIndex, all_letters

log = "Utils Module"


def meta():
    from collections import OrderedDict
    with open("dataset/meta.json", "r") as File:
        Dict = json.loads(File.read(), object_pairs_hook=OrderedDict)
    return Dict


def categoryFromOutput(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def saveList(List, path):
    with open(path, "w") as f:
        json.dump(List, f, indent=2)


class Data:

    def __init__(self, meta, data):
        self.data = data
        self.languages = meta["languages"]
        self.letters = meta["all_letters"]

    def randomChoice(self, List):
        return List[random.randint(0, len(List) - 1)]

    def randomTrainingExample(self, device):
        category = self.randomChoice(self.languages)
        line_tensor = self.randomChoice(self.data[category])
        line = self.tensorToLine(line_tensor)
        category_tensor = torch.tensor([self.languages.index(category)], dtype=torch.long, device=device)
        return category, line, category_tensor, line_tensor

    def generateTrainData(self, n_iter, device):
        for i in range(n_iter):
            yield (i, *self.randomTrainingExample(device))

    def tensorToLine(self, tensor):
        word = []
        for i in range(tensor.shape[0]):
            _, index = tensor[i, :].topk(1)
            index = index[0].item()
            word.append(all_letters[index])
        return "".join(word)

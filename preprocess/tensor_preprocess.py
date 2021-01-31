import json
from collections import OrderedDict
import torch
from text_preprocess import lineToTensor


log = "Dataset Creation"


def main():
    with open("../dataset/raw_data/raw_data.json", "r") as File:
        data = json.loads(File.read(), object_pairs_hook=OrderedDict)

    tensors = OrderedDict()
    catagories = data["meta"]["languages"]

    for lang in data["data"].keys():
        tensors[lang] = []
        for name in data["data"][lang]:
            tensors[lang].append(lineToTensor(name))
        print(f"Done {lang}")

    n = 0
    for i in tensors.values():
        n += len(i)
    print(n, data["meta"]["num_names"])
    torch.save(tensors, "../dataset/dataset.pt")

if __name__ == "__main__":
    print(log)
    main()

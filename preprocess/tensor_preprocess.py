import json
from collections import OrderedDict
import torch
from text_preprocess import lineToTensor, labelToTensor


log = "Dataset Creation"


def main():
    with open("../dataset/raw_data.json", "r") as File:
        data = json.loads(File.read(), object_pairs_hook=OrderedDict)

    tensors = OrderedDict()
    catagories = data["meta"]["languages"]

    for name in data["data"].keys():
        for i in range(len(data["data"][name])):
            tensors[labelToTensor(name, catagories)] = lineToTensor(data["data"][name][i])
        print(f"Done {name}")

    print(len(tensors.keys()), data["meta"]["num_names"])
    torch.save(tensors, "../dataset/dataset.pt")

if __name__ == "__main__":
    print(log)
    main()

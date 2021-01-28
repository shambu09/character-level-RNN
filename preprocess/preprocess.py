import os
import glob
import unicodedata
import string
import json
from collections import OrderedDict

log = "-------Data Preprocessing-------"

all_letters = string.ascii_letters + " .,;'"


def unicodeToAscii(s):
    n_letters = len(all_letters)

    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def readlines(filename):
    with open(filename, encoding="utf-8") as f:
        lines = list(map(unicodeToAscii, f.read().strip().split('\n')))
    return lines


def letterToTensor(letter):
    return all_letters.find(letter)


def main():
    directory = r"..\dataset\text_data\names\*"
    files = glob.glob(directory)

    data_dict = OrderedDict()
    meta_data = OrderedDict()
    data_set = OrderedDict()
    all_categories = []
    num_names = 0

    for File in files:
        category = os.path.splitext(os.path.basename(File))[0]
        names_buffer = readlines(File)
        num_names += len(names_buffer)
        data_dict[category] = names_buffer
        all_categories.append(category)

    meta_data["languages"] = all_categories
    meta_data["num_categories"] = len(all_categories)
    meta_data["num_names"] = num_names
    data_set["meta"] = meta_data
    data_set["data"] = data_dict

    with open(r"..\dataset\raw_data.json", "w") as File:
        File.write(json.dumps(data_set))

    with open(r"..\dataset\raw_data_formatted.json", "w") as File:
        File.write(json.dumps(data_set, indent=6))


if __name__ == "__main__":
    print(log)
    main()

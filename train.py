from models.model import RNN
from utils import meta, lineToTensor, labelToTensor
import string
import torch

meta_data = meta()
num_hidden = 128
num_letters = meta_data["num_letters"]
num_categories = meta_data["num_categories"]

net = RNN(num_letters, num_hidden, num_categories)

input_tensor = lineToTensor("Albert")
hidden = net.initHidden()

output = net(input_tensor[0], hidden)
print(output)

import time
import string
import random
import torch
from models.model import RNN
from utils import meta, lineToTensor, categoryFromOutput, timeSince, Data, saveList
from collections import OrderedDict

log = "Training"
meta_data = meta()
num_hidden = 128
num_letters = meta_data["num_letters"]
num_categories = meta_data["num_categories"]
languages = meta_data["languages"]
num_names = meta_data["num_names"]

# GPU
device = torch.device("cuda:0")

# Defining Model and loss function
rnn = RNN(num_letters, num_hidden, num_categories)
rnn.to(device)

criterion = torch.nn.NLLLoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

# Train block for a single word.


def train(label_tensor, line_tensor):
    hidden = rnn.initHidden().to(device)
    rnn.zero_grad()

    for i in range(line_tensor.shape[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, label_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.item()


def main():
    dataset = torch.load("dataset/dataset.pt", map_location=device)
    data = Data(meta_data, dataset)

    n_iters = 100000
    print_every = 5000
    plot_every = 1000
    gen_data = data.generateTrainData(n_iters, device)

    current_loss = 0
    all_losses = []
    summary = OrderedDict()
    start = time.time()

    for train_data in gen_data:
        iter_n, category, line, category_tensor, line_tensor = train_data
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

        if (iter_n + 1) % print_every == 0:
            guess, guess_i = categoryFromOutput(output, data.languages)
            correct = f"Correct!" if category == guess else f"Wrong, guessed :{guess}"
            print(f"iterations: {iter_n}, {int(iter_n/n_iters) * 100}% Completed, time:{timeSince(start)}, loss: {loss:0.6f}, name: {line}, Origin: {category}, guess: {correct}")

        if iter_n % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    torch.save(rnn.state_dict(), "weights/weights.pth")
    saveList(all_losses, "eval/losses.json")

if __name__ == "__main__":
    print(log)
    main()

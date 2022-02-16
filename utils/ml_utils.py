import os

import torch
from torch.autograd import Variable


def train(train_dataloader, test_dataloader, batch_size, model, optimizer, criterion, learning_rate, epochs, device):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    total_batch = len(train_dataloader.dataset) // batch_size

    for epoch in range(epochs):
        avg_cost = 0
        epoch_loss = []
        epoch_acc = []

        for i, batch in enumerate(train_dataloader):
            batch_X, batch_Y = (b.to(device) for b in batch)
            X, Y = Variable(batch_X), Variable(batch_Y)

            optimizer.zero_grad()

            hypothesis = model(X)
            cost = criterion(hypothesis, Y)

            cost.backward()
            optimizer.step()

            prediction = hypothesis.data.max(dim=1)[1]
            epoch_acc.append(((prediction.data == Y.data).float().mean()).item())
            epoch_loss.append(cost.item())

            if i % 200 == 0:
                print("Epoch= {},\t batch = {},\t cost = {:2.4f},\t accuracy = {}".format(epoch + 1, i, epoch_loss[-1],
                                                                                          epoch_acc[-1]))

            avg_cost += cost.data / total_batch
        train_loss.append(sum(epoch_loss) / len(epoch_loss))
        train_acc.append(sum(epoch_acc) / len(epoch_acc))
        test_1, test_2 = test(test_dataloader, model, criterion, device)
        test_loss.append(test_1)
        test_acc.append(test_2)
        print("[Epoch: {:>4}], averaged cost = {:>.9}".format(epoch + 1, avg_cost.item()))

    path = f'tmp/{type(model).__name__}'
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, f'model_{batch_size}_{learning_rate}.pth'))
    print(f'Learning finished with batch size {batch_size} and lr {learning_rate}')
    return {"train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc}


def test(test_dataloader, model, criterion, device):
    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            batch_X, batch_Y = (b.to(device) for b in batch)
            X, Y = Variable(batch_X), Variable(batch_Y)

            outputs = model(X)

            hypothesis = model(X)
            cost = criterion(hypothesis, Y)
            running_loss += cost.item()

            _, predicted = outputs.max(1)
            total += Y.size(0)
            correct += predicted.eq(Y).sum().item()

    test_loss = running_loss / len(test_dataloader)
    accu = 100. * correct / total

    return test_loss, accu

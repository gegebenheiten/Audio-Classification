import pdb

import torch.nn as nn
import torch


def evaluate(model, device, criterion, test_loader, model_name):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        # pdb.set_trace()
        for data, target in test_loader:
            batch_count += 1
            data = torch.tensor(data, dtype=torch.float).unsqueeze(1)
            if model_name == 'mobilenet':
                data = data.repeat(1, 3, 1, 1).to(device)
            else:
                data = data.to(device)
            target = target.to(device)

            # Get the predicted classes for this batch
            output = model(data)

            # Calculate the loss for this batch
            test_loss += criterion(output, target).item()

            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        accuracy))

    # return average loss for the epoch
    return avg_loss, accuracy

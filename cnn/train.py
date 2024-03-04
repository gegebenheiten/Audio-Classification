import pdb
import random
import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import warnings
from model import CNNModel
from dataloader import AudioDataset
from evaluate import evaluate
from tqdm import tqdm
import importlib.util
from mobilenet import MobileNetV2
from torch.optim.lr_scheduler import StepLR

warnings.filterwarnings('ignore')


def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, model_name):
    model.train()
    train_loss = 0
    print("------------------------------- Epoch:", epoch, "-------------------------------")
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # pdb.set_trace()
        data = torch.tensor(data, dtype=torch.float).unsqueeze(1)
        if model_name == 'mobilenet':
            data = data.repeat(1, 3, 1, 1).to(device)
            # pdb.set_trace()
        else:
            data = data.to(device)
        target = target.to(device)

        # Reset the optimizer
        optimizer.zero_grad()

        # Push the data forward through the model layers
        output = model(data.to(device))

        # Get the loss
        loss = criterion(output, target)

        # Keep a running total
        train_loss += loss.item()

        # Backpropagation
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

    avg_loss = train_loss / (batch_idx + 1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss


def training(model, device, train_loader, test_loader, criterion, optimizer, scheduler, epoch, model_name):
    # Track metrics in these arrays
    epoch_nums = []
    training_loss = []
    validation_loss = []
    validation_accuracy = []

    epochs = epoch
    print('Training on', device)
    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = train(model, device, train_loader, criterion, optimizer, scheduler, epoch, model_name)
        test_loss, test_acc = evaluate(model, device, criterion, test_loader, model_name)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)
        validation_accuracy.append(test_acc)

    # # Plot the training and validation loss
    # plt.figure(figsize=(10, 5))
    # plt.plot(epoch_nums, training_loss, 'r-', label='Training Loss')
    # plt.plot(epoch_nums, validation_loss, 'b-', label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title('Training and Validation Loss Over Epochs')
    # plt.savefig(f'/home/lucas/ESC/cnn/model_only_on_esc_dataset/figures/{model_name}_{epochs}_training_validation_loss.png')  # Save the plot
    # plt.show()

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epoch_nums, training_loss, 'r-', label='Training Loss')
    ax1.plot(epoch_nums, validation_loss, 'b-', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color)  # 设置第二个y轴的标签
    ax2.plot(epoch_nums, validation_accuracy, 'g-', label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Training and Validation Loss/Accuracy')
    plt.savefig(
        f'/home/lucas/ESC/cnn/model_only_on_esc_dataset/figures/{model_name}_{epochs}_training_validation_loss_accuracy.png')  # Save the plot
    plt.show()

    # Save the model's state dictionary
    model_save_path = f'/home/lucas/ESC/cnn/model_only_on_esc_dataset/{model_name}_{epochs}.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')


def load_features(data_loader):
    for i, (features, category) in enumerate(data_loader):
        print(f"Batch {i + 1}")
        print(features.shape, category)


def main(model_name='cnn'):
    random.seed(20240215)
    audio_dir = '/home/lucas/ESC/data'
    with open('/home/lucas/ESC/cnn/config.json', 'r') as f:
        config = json.load(f)

    dataset = AudioDataset(audio_dir=audio_dir, config=config)

    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    load_features(train_loader)

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    load_features(test_loader)

    device = "cpu"
    if torch.cuda.is_available():
        # if GPU available, use cuda (on a cpu, training will take a considerable length of time!)
        device = "cuda"

    # pdb.set_trace()
    if model_name == "cnn":
        model = CNNModel(num_classes=12).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        pdb.set_trace()

    elif model_name == 'mobilenet':
        model = MobileNetV2(num_classes=12)
        # pdb.set_trace()
        model_weight_path = "./mobilenet_v2.pth"
        assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
        pre_weights = torch.load(model_weight_path, map_location='cpu')

        # delete classifier weights
        pre_dict = {k: v for k, v in pre_weights.items() if model.state_dict()[k].numel() == v.numel()}
        missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)

        # freeze features weights
        for param in model.features.parameters():
            param.requires_grad = False

        model.to(device)

        # define loss function
        criterion = nn.CrossEntropyLoss()

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=config['lr'])
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        raise ValueError("unknown model: {}".format(model_name))
    # pdb.set_trace()
    training(model, device, train_loader, test_loader, criterion, optimizer, scheduler, config['epoch'], model_name)


if __name__ == '__main__':
    model_selected = 'mobilenet'  # mobilenet
    main(model_selected)

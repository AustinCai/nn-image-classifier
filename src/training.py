import torch
from torch import nn 
import torch.nn.functional as F
import time
from pathlib import Path
import datetime

import visualize
import models

from util import Constants
from util import Objects

# def seed(s):
#     torch.manual_seed(s)

def model_wrapper(model, x, label_str=None): 
    yh = model(x)
    predictions = torch.argmax(yh, dim=1)

    if label_str:
        if not label_str == "":
            label_str = "_" + label_str
        print("    x{}.size(): {}".format(label_str, x.size()))   
        print("    yh{}.size(): {}".format(label_str, yh.size()))
        print("    predictions.size(): {}".format(predictions.size()))
    return yh, predictions


def run_batch(model, loss_func, x_batch, y_batch, optimizer=None, verbose=False):
    yh_batch, predictions = model_wrapper(model, x_batch)

    accuracy = (predictions == y_batch).float().mean()
    loss = loss_func(yh_batch, y_batch)

    if optimizer: # perform learning
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return accuracy, loss.item()


def run_epoch(model, loss_func, dataloader, 
    optimizer=None, validation_dataloader=None, training_statistics_arr=None, verbose=False, fast=False):
    running_loss, running_accuracy = 0.0, 0.0

    for i, (x_batch, y_batch) in enumerate(dataloader):
        if i > 10 and fast:
            break
        accuracy, loss = run_batch(model, loss_func, x_batch, y_batch, optimizer, verbose)
        running_accuracy += accuracy
        running_loss += loss

    epoch_accuracy, epoch_loss = running_accuracy/len(dataloader), running_loss/len(dataloader)
    
    if validation_dataloader:
        epoch_validation_accuracy, epoch_validation_loss = run_epoch(model, loss_func, validation_dataloader)

    if training_statistics_arr:
        training_statistics_arr[-1]["loss"] = epoch_loss
        training_statistics_arr[-1]["accuracy"] = epoch_accuracy.item()
        training_statistics_arr.append({"loss": None, "accuracy": None,
            "validation_loss": epoch_validation_loss, 
            "validation_accuracy": epoch_validation_accuracy.item()
            })

        print("    Epoch {}: train acc {}, val acc {} || train loss {}, val loss {}.".format(len(training_statistics_arr)-1,
            training_statistics_arr[-2]["accuracy"], training_statistics_arr[-2]["validation_accuracy"], 
            training_statistics_arr[-2]["loss"], training_statistics_arr[-2]["validation_loss"]))

    return epoch_accuracy, epoch_loss


def run_training(model, loss_func, dataloader, validation_dataloader, optimizer, args):
    start_time = time.time()

    '''
    Validation Offset:
    Validation statistics (validation accuracy and loss) V_e for epoch e is defined as the statistics the model achives 
    over the validation set after completing e-1 epochs of training. Training statistics (training accuracy and loss) 
    T_e for epoch e are defined as the averaged statistics over all training batches in epoch e. Eg. the validation 
    accuracy and loss of epoch 1 is 0 and NA, and all future valued are offset by one. Without this shift, the model 
    that validation statistics is calculated over would have undergone more training than the model that training 
    statistics is calculated over, artificially inflating validation values. 
    '''
    training_statistics_arr = [{
        "loss": None, # will be replaced, implementing the validation offset
        "accuracy": None, # will be replaced, implementing the validation offset
        "validation_loss": "NA", # val loss of epoch 1 defined as NA
        "validation_accuracy": "NA" # val accuracy of epoch 1 defined as NA
        }]

    for epoch in range(args.epochs):
        run_epoch(model, loss_func, dataloader, 
            optimizer=optimizer, validation_dataloader=validation_dataloader, 
            training_statistics_arr=training_statistics_arr, verbose=args.verbose, fast=args.fast)
    training_statistics_arr.pop() # get rid of last element, which has loss and accuracy values unset because of validation offset

    print("Training completed in {} seconds.".format(time.time() - start_time))

    return training_statistics_arr


def init_optimizer(model):
    if Constants.optimizer_str == "sgd":
        return torch.optim.SGD(model.parameters(), lr=Constants.learning_rate)
    if Constants.optimizer_str == "adam":
        return torch.optim.Adam(model.parameters(), lr=Constants.learning_rate)
    raise Exception("Invalid optimizer specification of {}.".format(optimizer_str))


def init_model():
    in_channels = 784 if Constants.dataset_str == "mnist" else 3072

    if Constants.model_str == "linear": 
        return models.Linear(in_channels, Constants.out_channels).to(Objects.dev)

    if Constants.model_str == "small_nn": 
        return models.SmallNN(in_channels, Constants.out_channels).to(Objects.dev)
    if Constants.model_str == "large_nn":
        return models.LargeNN(in_channels, Constants.out_channels).to(Objects.dev)

    if Constants.model_str == "small_cnn":
        return models.SmallCNN(Constants.out_channels).to(Objects.dev)
    if Constants.model_str == "best_cnn":
        return models.BestCNN(Constants.out_channels).to(Objects.dev)

    raise Exception("Invalid model_str specification of {}.".format(Constants.model_str))


def save_model(model, optimizer, loss, save_str, args):

    torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, Path("models") / save_str)


def load_model(path, model, optimizer):
    saved_model_dict = torch.load(Path(__file__).parent.resolve() / "../{}".format(path))
    model.load_state_dict(saved_model_dict['model_state_dict'])
    optimizer.load_state_dict(saved_model_dict['optimizer_state_dict'])
    epoch = saved_model_dict['epoch']
    loss = saved_model_dict['loss']

    return model, optimizer, epoch, loss
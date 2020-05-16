import torch
from torch import nn 
import torch.nn.functional as F
import time

import visualize
import models

from util import Constants

# def seed(s):
#     torch.manual_seed(s)

def model_wrapper(label_str, model, x):
    if label_str:
        print("    x_{}.size(): {}".format(label_str, x.size()))
        print("    y_{}.size(): {}".format(label_str, y.size()))       
    yh = model(x)
    if label_str:
        print("    yh_{}.size(): {}".format(label_str, yh.size()))

    predictions = torch.argmax(yh, dim=1)
    return yh, predictions


def run_batch(model, loss_func, x_batch, y_batch, epoch=None, optimizer=None, verbose=False):
    # print only once, across all batches and epochs
    label_str = "batch" if (verbose and not i and not epoch) else None
    yh_batch, predictions = model_wrapper(label_str, model, x_batch)
    accuracy = (predictions == y_batch).float().mean()

    loss = loss_func(yh_batch, y_batch)

    if optimizer: # perform learning
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return accuracy, loss.item()


def run_epoch(model, loss_func, dataloader, epoch=None, optimizer=None, validation_dataloader=None, training_statistics_arr=None, verbose=False):
    running_loss, running_accuracy = 0.0, 0.0

    for i, (x_batch, y_batch) in enumerate(dataloader):
        accuracy, loss = run_batch(model, loss_func, x_batch, y_batch, epoch, optimizer, verbose)
        running_accuracy += accuracy
        running_loss += loss

    epoch_accuracy, epoch_loss = running_accuracy/len(dataloader), running_loss/len(dataloader)
    
    if validation_dataloader:
        epoch_validation_accuracy, epoch_validation_loss = run_epoch(model, loss_func, validation_dataloader)

    if training_statistics_arr and epoch != None:
        training_statistics_arr[-1]["loss"] = epoch_loss
        training_statistics_arr[-1]["accuracy"] = epoch_accuracy.item()
        training_statistics_arr.append({"loss": None, "accuracy": None,
            "validation_loss": epoch_validation_loss, 
            "validation_accuracy": epoch_validation_accuracy.item()
            })

        print("    Epoch {}: train acc {}, val acc {} || train loss {}, val loss {}.".format(epoch+1,
            training_statistics_arr[-2]["accuracy"], training_statistics_arr[-2]["validation_accuracy"], 
            training_statistics_arr[-2]["loss"], training_statistics_arr[-2]["validation_loss"]))

    return epoch_accuracy, epoch_loss


def run_training(model, loss_func, dataloader, validation_dataloader, optimizer, writer, run_spec, verbose=False):
    start_time = time.time()
    print('Training {} model with a \'{}\' optimization and \'{}\' augmentation over {} epochs'.format(
        run_spec["model_str"], run_spec["optimizer"], run_spec["augmentation"], run_spec["epochs"]))

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
        "validation_accuracy": 0.0 # val accuracy of epoch 1 defined as 0
        }]

    for epoch in range(run_spec["epochs"]):
        run_epoch(model, loss_func, dataloader, epoch, optimizer, validation_dataloader, training_statistics_arr, verbose)
    training_statistics_arr.pop() # get rid of last element, which has loss and accuracy values unset because of validation offset

    write_training_statistics(writer, training_statistics_arr)
    writer.close()
    print("Training completed in {} seconds.".format(time.time() - start_time))

    return training_statistics_arr


def write_training_statistics(writer, training_statistics_arr):
    # This loop exclude the last entry training_statistics_arr, which has undefined train accuracy and loss.
    # See validation offset comment. 
    for epoch in range(len(training_statistics_arr)-1): 
        epoch_stats = training_statistics_arr[epoch]

        writer.add_scalars(
            "Validation vs. Train Accuracy", 
            {"validation": epoch_stats["validation_accuracy"], "train": epoch_stats["accuracy"]}, 
            global_step=epoch+1)
        writer.add_scalar('Accuracy/Train', epoch_stats["accuracy"], global_step=epoch+1)
        writer.add_scalar('Accuracy/Validation', epoch_stats["validation_accuracy"], global_step=epoch+1)
        writer.add_scalar('Loss/Train', epoch_stats["loss"], global_step=epoch+1)   
        if (epoch): # validation of first epoch is undefined
            writer.add_scalar('Loss/Validation', epoch_stats["validation_loss"], global_step=epoch+1) 


def init_optimizer(run_spec, model):
    if run_spec["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=run_spec["lr"])
    if run_spec["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=run_spec["lr"])
    raise Exception("Invalid optimizer specification of {}.".format(run_spec["optimizer"]))


def init_model(model_type, dataset, dev):
    in_channels = 784 if dataset == "mnist" else 3072
    out_channels = 10

    if model_type == "linear": 
        return models.Linear(in_channels, out_channels).to(dev)

    if model_type == "small_nn": 
        return models.SmallNN(in_channels, out_channels).to(dev)
    if model_type == "large_nn":
        return models.LargeNN(in_channels, out_channels).to(dev)

    if model_type == "small_cnn":
        return models.SmallCNN(out_channels).to(dev)
    if model_type == "best_cnn":
        return models.BestCNN(out_channels).to(dev)

    raise Exception("Invalid model_str specification of {}.".format(model_type))
import torch
from torch import nn 
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import time
import datetime
import random

import visualize


#model
class Mnist_Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, xb):
        return self.lin(xb)


class Mnist_NN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        IN_HIDDEN, OUT_HIDDEN = 32, 32
        self.l1 = nn.Linear(in_channels, IN_HIDDEN)
        self.l2 = nn.Linear(IN_HIDDEN, OUT_HIDDEN)
        self.l3 = nn.Linear(OUT_HIDDEN, out_channels) 

    def forward(self, xb):
        a1 = F.relu(self.l1(xb))
        a2 = F.relu(self.l2(a1))
        return self.l3(a2)


def accuracy(yh_batch, y_batch):
    '''Caclulates the model's per-batch accuracy given its 
    batch_size x class_count) output and (batch_size) labels. Does this by
    interpreting the models prediction on each data point as the 
    highest-scored class.
    '''
    preds = torch.argmax(yh_batch, dim=1)
    return (preds == y_batch).float().mean()


def fit_model(writer, run_spec, model, train_dlr, loss_func, optimizer, valid_dlr, batches_between_log=10):
    '''Trains the model. For each epoch, iterates through all batches, 
    calculating loss and optimizing model weights after each batch. The 
    running loss is also saved to tensorboard. 
    '''
    print('Training {} model with a \'{}\' optimization and \'{}\' augmentation over {} epochs'.format(
        run_spec["model_str"], run_spec["optimizer"], run_spec["augmentation"], run_spec["epochs"]))
    
    start_time = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    for epoch in range(run_spec["epochs"]):
        for i, (x_batch, y_batch) in enumerate(train_dlr):
            yh_batch = model(x_batch)

            loss = loss_func(yh_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += accuracy(yh_batch, y_batch)
            
            # Prints loss and accuracy info at the end of each epoch. 
            if i == len(train_dlr) - 1:              
                with torch.no_grad():
                    x_valid, y_valid = iter(valid_dlr).__next__()
                    yh_valid = model(x_valid)   
                    valid_accuracy = accuracy(yh_valid, y_valid)  

                writer.add_scalar('Validation Accuracy (epoch)', 
                                  valid_accuracy, 
                                  global_step=epoch * len(train_dlr) + i)
                print("    epoch {}: loss {}, validation accuracy {}, train accuracy {}.".format(
                    epoch+1, loss.item(), valid_accuracy, accuracy(yh_batch, y_batch)))
                
            # Logs loss and accuracy info to tensorboard every 10 batches.
            if i % batches_between_log == batches_between_log-1:
                writer.add_scalar('Training Loss (batch)', 
                                  running_loss/batches_between_log, 
                                  global_step=epoch * len(train_dlr) + i)
                writer.add_scalar('Training Accuracy (batch)', 
                                  running_accuracy/batches_between_log, 
                                  global_step=epoch * len(train_dlr) + i)
                running_loss = 0.0
                running_accuracy = 0.0

    writer.close()
    print("Training completed in {} seconds.".format(time.time() - start_time))


def eval_model(writer, test_dlr, model, run_spec, verbose):
    '''Runs the model over the test set, saving the model's accuracy alongside
    its hyperparameters to tensorboard. 
    '''
    test_dlr_iter = iter(test_dlr)
    x_batch_test, y_batch_test = test_dlr_iter.__next__()

    yh_batch_test = model(x_batch_test)
    x_batch_test = x_batch_test.detach().cpu()
    y_batch_test = y_batch_test.detach().cpu()
    yh_batch_test = yh_batch_test.detach().cpu()

    accuracy_test = accuracy(yh_batch_test, y_batch_test)
    print("Final model test accuracy: {}.".format(accuracy_test.item()))

    writer.add_hparams(run_spec, {'accuracy': accuracy_test})
    yh_batch_test = torch.argmax(yh_batch_test, dim=1) 
    visualize.print_save_cmatrix(writer, y_batch_test, yh_batch_test)



def init_model(model_type, dataset, dev):
    MNIST_IN_CHANNELS, CIFAR10_IN_CHANNELS = 784, 3072
    in_channels = MNIST_IN_CHANNELS if dataset == "mnist" else CIFAR10_IN_CHANNELS
    out_channels = 10

    if model_type == "nn": 
        return Mnist_NN(in_channels, out_channels).to(dev)
    if model_type == "linear": 
        return Mnist_Linear(in_channels, out_channels).to(dev)
    return "ERROR"
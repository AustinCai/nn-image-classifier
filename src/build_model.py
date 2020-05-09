import torch
from torch import nn 
import torch.nn.functional as F
import time

import visualize
import models

# def seed(s):
#     torch.manual_seed(s)

def accuracy(yh, y):
    '''Caclulates the model's per-batch accuracy given its 
    batch_size x class_count) output and (batch_size) labels. Does this by
    interpreting the models prediction on each data point as the 
    highest-scored class.
    '''
    preds = torch.argmax(yh, dim=1)
    accuracy = (preds == y).float().mean()
    return accuracy


def predict_with_logs(label_str, model, x, y, print_logs):
    if print_logs:
        print("    x_{}.size(): {}".format(label_str, x.size()))
        print("    y_{}.size(): {}".format(label_str, y.size()))       
    yh = model(x)
    if print_logs:
        print("    yh_{}.size(): {}".format(label_str, yh.size()))
    return yh


def log_epoch_statistics(writer, model, valid_dl, loss, train_accuracy, epoch, verbose):
    if not epoch:
        print("    Note: Validation accuracy is calculated on the model once it has been trained over the entire epoch, \n"
            + "    while training accuracy is calculated while the model is still training within the epoch. \n"
            + "    Thus, validation accuracy may be higher than training accuracy when the model is learning quickly \n" \
            + "    (eg. during the first epochs).")

    with torch.no_grad():
        x_valid, y_valid = iter(valid_dl).__next__()

        print_logs = verbose and not epoch # prints once per epoch
        yh_valid = predict_with_logs("valid", model, x_valid, y_valid, print_logs)

        validation_accuracy = accuracy(yh_valid, y_valid)

    writer.add_scalars("Validation vs. Train Accuracy", {"validation": validation_accuracy, "train": train_accuracy}, global_step=epoch)
    writer.add_scalar('Training Accuracy', train_accuracy, global_step=epoch)
    writer.add_scalar('Validation Accuracy', validation_accuracy, global_step=epoch)
    writer.add_scalar('Loss', loss, global_step=epoch)
    print("    EPOCH {}: loss {}, val acc {}, train acc {}.".format(epoch+1, loss, validation_accuracy, train_accuracy))


def fit_model(writer, run_spec, model, train_dl, loss_func, optimizer, valid_dl, verbose):
    '''Trains the model. For each epoch, iterates through all batches, 
    calculating loss and optimizing model weights after each batch. The 
    running loss is also saved to tensorboard. 
    '''
    print('BEGIN TRAINING: {} model with a \'{}\' optimization and \'{}\' augmentation over {} epochs'.format(
        run_spec["model_str"], run_spec["optimizer"], run_spec["augmentation"], run_spec["epochs"]))


    start_time = time.time()

    for epoch in range(run_spec["epochs"]):
        running_loss, running_train_accuracy = 0.0, 0.0

        for i, (x_batch, y_batch) in enumerate(train_dl):

            print_logs = verbose and not i and not epoch # print only once, across all batches and epochs 
            yh_batch = predict_with_logs("batch", model, x_batch, y_batch, print_logs)

            loss = loss_func(yh_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_train_accuracy += accuracy(yh_batch, y_batch)
            
        # Prints loss and accuracy info from the last batch of each epoch.              
        log_epoch_statistics(writer, model, valid_dl, running_loss/i, running_train_accuracy/i, epoch, verbose)
      
    writer.close()
    print("Training completed in {} seconds.".format(time.time() - start_time))


def eval_model(writer, test_dl, model, run_spec, verbose):
    '''Runs the model over the test set, saving the model's accuracy alongside
    its hyperparameters to tensorboard. 
    '''
    x_batch_test, y_batch_test = iter(test_dl).__next__()
    yh_batch_test = model(x_batch_test)

    accuracy_test = accuracy(yh_batch_test, y_batch_test)
    print("Final model test accuracy: {}.".format(accuracy_test.item()))

    writer.add_hparams(run_spec, {'accuracy': accuracy_test})
    preds = torch.argmax(yh_batch_test, dim=1) 
    visualize.print_save_cmatrix(writer, y_batch_test.detach().cpu(), preds.detach().cpu())


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
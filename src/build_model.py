import torch
from torch import nn 
import torch.nn.functional as F
import time

import visualize
import models

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


# def eval_model(model, loss_func, dataloader, verbose=False):
#     return train_and_eval_model(model, loss_func, dataloader, verbose=False)

def run_epoch(model, loss_func, dataloader, optimizer=None, validation_dataloader=None, training_statistics_arr=None, verbose=False):
    running_loss, running_accuracy = 0.0, 0.0

    for i, (x_batch, y_batch) in enumerate(dataloader):
        if i > 10: break

        # print only once, across all batches and epochs
        label_str = "batch" if (verbose and not i and not epoch) else None
        yh_batch, predictions = model_wrapper(label_str, model, x_batch)
        accuracy = (predictions == y_batch).float().mean()

        loss = loss_func(yh_batch, y_batch)

        if optimizer: # perform learning
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_accuracy += accuracy
        running_loss += loss.item()

    epoch_accuracy, epoch_loss = running_accuracy/len(dataloader), running_loss/len(dataloader)

    # if evaluating (not learning > optimizer == None), print
    # if not optimizer: # without learning, function execution ends here ------------------------------------
    #     return epoch_accuracy, epoch_loss
    
    if optimizer:
        epoch_validation_accuracy, epoch_validation_loss = run_epoch(model, loss_func, validation_dataloader)

        training_statistics_arr[-1]["loss"] = epoch_loss
        training_statistics_arr[-1]["accuracy"] = epoch_accuracy.item()
        training_statistics_arr.append({
            "loss": None,
            "accuracy": None,
            "validation_loss": epoch_validation_loss,
            "validation_accuracy": epoch_validation_accuracy.item()
            })
        # loss_arr.append(epoch_loss)
        # accuracy_arr.append(epoch_accuracy)
        # validation_loss_arr.append(epoch_validation_loss)
        # validation_accuracy_arr.append(epoch_validation_accuracy)
        print("    EPOCH: train acc {}, val acc {} || train loss {}, val loss {}.".format( 
            training_statistics_arr[-2]["accuracy"], training_statistics_arr[-2]["validation_accuracy"], 
            training_statistics_arr[-2]["loss"], training_statistics_arr[-2]["validation_loss"]))

    return epoch_accuracy, epoch_loss


def train_and_eval_model(model, loss_func, dataloader, 
    validation_dataloader=None, optimizer=None, writer=None, run_spec=None, verbose=False):
    if optimizer:
        start_time = time.time()
        print('BEGIN TRAINING: {} model with a \'{}\' optimization and \'{}\' augmentation over {} epochs'.format(
            run_spec["model_str"], run_spec["optimizer"], run_spec["augmentation"], run_spec["epochs"]))

        # Validation accuracy V_e for epoch e is defined as the accuracy the model achives over the validation set 
        # after completing e-1 epochs of training. The loss L_e and train accuracy T_e for epoch e are defined as
        # the averaged loss and accuracy over all training batches in epoch e. Eg. the validation accuracy of epoch 
        # 1 is 0, and all future valued are offset by one. Without this shift, the model that validation accuracy is
        # calculated over would have undergone more training, artificially inflatin its value. 
        # accuracy_arr, loss_arr, validation_accuracy_arr, validation_loss_arr = [], [], [0], ["NA"]
    training_statistics_arr = [{
        "loss": None, # will be replaced
        "accuracy": None, # will be replaced
        "validation_loss": "NA", # val loss of epoch 1 defined as NA
        "validation_accuracy": 0 # val accuracy of epoch 1 defined as 0
        }]

    # returns after first epoch if not learning
    epochs = run_spec["epochs"] if optimizer else 1
    for epoch in range(epochs):
        run_epoch(model, loss_func, dataloader, optimizer, validation_dataloader, training_statistics_arr, verbose)
        # running_loss, running_accuracy = 0.0, 0.0

        # for i, (x_batch, y_batch) in enumerate(dataloader):
        #     # print only once, across all batches and epochs
        #     label_str = "batch" if (verbose and not i and not epoch) else None
        #     yh_batch, predictions = model_wrapper(label_str, model, x_batch)
        #     accuracy = (predictions == y_batch).float().mean()

        #     loss = loss_func(yh_batch, y_batch)

        #     if optimizer: # perform learning
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

        #     running_accuracy += accuracy
        #     running_loss += loss.item()

        # epoch_accuracy, epoch_loss = running_accuracy/len(dataloader), running_loss/len(dataloader)

        # # if evaluating (not learning > optimizer == None), print
        # if not optimizer: # without learning, function execution ends here ------------------------------------
        #     return epoch_accuracy, epoch_loss
        
        # epoch_validation_accuracy, epoch_validation_loss = eval_model(model, loss_func, validation_dataloader)

        # loss_arr.append(epoch_loss)
        # accuracy_arr.append(epoch_accuracy)
        # validation_loss_arr.append(epoch_validation_loss)
        # validation_accuracy_arr.append(epoch_validation_accuracy)

        # print("    EPOCH {}: train acc {}, val acc {} || train loss {}, val loss {}.".format(
        #     epoch+1, accuracy_arr[-1], validation_accuracy_arr[-2], loss_arr[-1], validation_loss_arr[-2]))
    

    # TODO - FINISH write_run_plots
    if optimizer:
        # write_training_statistics(writer, training_statistics_arr)
        # write_run_plots(writer, loss_arr, accuracy_arr, validation_accuracy_arr, validation_loss_arr)
        writer.close()
        print("Training completed in {} seconds.".format(time.time() - start_time))

def write_training_statistics(writer, training_statistics_arr):
    for epoch, epoch_stats in enumerate(training_statistics_arr):
        writer.add_scalars(
            "Validation vs. Train Accuracy", 
            {"validation": epoch_stats["validation_accuracy"], "train": epoch_stats["accuracy"]}, 
            global_step=epoch+1)
        writer.add_scalar('Accuracy/Train', epoch_stats["accuracy"], global_step=epoch+1)
        writer.add_scalar('Accuracy/Validation', epoch_stats["validation_accuracy"], global_step=epoch+1)
        writer.add_scalar('Loss/Train', epoch_stats["loss"], global_step=epoch+1)   
        if (epoch): # validation of first epoch is undefined
            writer.add_scalar('Loss/Validation', epoch_stats["validation_loss"], global_step=epoch+1) 


def write_run_plots(writer, train_loss_arr, train_accuracy_arr, validation_accuracy_arr, validation_loss_arr):
    # This for loop skips the last entry of validation_accuracy_arr, since the entire array is shifted over by one. 
    # See the comment above the initialization of validation_accuracy_arr.
    for epoch in range(len(train_loss_arr)):
        writer.add_scalars("Validation vs. Train Accuracy", 
            {"validation": validation_accuracy_arr[epoch], "train": train_accuracy_arr[epoch]}, global_step=epoch+1)
        writer.add_scalar('Accuracy/Train', train_accuracy_arr[epoch], global_step=epoch+1)
        writer.add_scalar('Accuracy/Validation', validation_accuracy_arr[epoch], global_step=epoch+1)
        writer.add_scalar('Loss/Train', train_loss_arr[epoch], global_step=epoch+1)   
        if (epoch):
            writer.add_scalar('Loss/Validation', validation_loss_arr[epoch], global_step=epoch+1) 


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
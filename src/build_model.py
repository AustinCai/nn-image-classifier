import torch
from torch import nn 
import torch.nn.functional as F
import time

import visualize
import models

# def seed(s):
#     torch.manual_seed(s)

def model_wrapper_2(label_str, model, x):
    if label_str:
        print("    x_{}.size(): {}".format(label_str, x.size()))
        print("    y_{}.size(): {}".format(label_str, y.size()))       
    yh = model(x)
    if label_str:
        print("    yh_{}.size(): {}".format(label_str, yh.size()))

    predictions = torch.argmax(yh, dim=1)
    return yh, predictions

# def model_wrapper(label_str, model, loss_func, x, y):
#     if label_str:
#         print("    x_{}.size(): {}".format(label_str, x.size()))
#         print("    y_{}.size(): {}".format(label_str, y.size()))       
#     yh = model(x)
#     if label_str:
#         print("    yh_{}.size(): {}".format(label_str, yh.size()))

#     predictions = torch.argmax(yh, dim=1)
#     accuracy = (predictions == y).float().mean()
#     loss = loss_func(yh, y)
#     return yh, predictions, accuracy, loss


# def eval_model(writer, dl, model, loss_func, run_spec):
#     '''Runs the model over the test set, saving the model's accuracy alongside
#     its hyperparameters to tensorboard. If writer is None, results will not be
#     logged and saved. 
#     '''
#     x_batch, y_batch = iter(dl).__next__()
#     yh_batch, predictions, accuracy, loss = model_wrapper(None, model, loss_func, x_batch, y_batch)

#     if (writer):
#         print("Final model test accuracy: {}.".format(accuracy.item()))
#         writer.add_hparams(run_spec, {'accuracy': accuracy})
#         visualize.print_save_cmatrix(writer, y_batch.detach().cpu(), predictions.detach().cpu())

#     return accuracy, loss

# if optimizer, also learn

def eval_model(model, loss_func, dataloader, verbose=False):
    return train_and_eval_model(model, loss_func, dataloader, verbose=False)

def train_and_eval_model(model, loss_func, dataloader, 
    validation_dataloader=None, optimizer=None, writer=None, run_spec=None, verbose=False):
    if optimizer:
        start_time = time.time()
        print('BEGIN TRAINING: {} model with a \'{}\' optimization and \'{}\' augmentation over {} epochs'.format(
            run_spec["model_str"], run_spec["optimizer"], run_spec["augmentation"], run_spec["epochs"]))

        accuracy_arr, loss_arr, validation_accuracy_arr, validation_loss_arr = [], [], [0], ["NA"]

    # returns after first epoch if not learning
    epochs = run_spec["epochs"] if optimizer else 1
    for epoch in range(epochs):
        running_loss, running_accuracy = 0.0, 0.0

        for i, (x_batch, y_batch) in enumerate(dataloader):

            label_str = "batch" if (verbose and not i and not epoch) else None
            yh_batch, predictions = model_wrapper_2(label_str, model, x_batch)
            accuracy = (predictions == y_batch).float().mean()

            loss = loss_func(yh_batch, y_batch)

            if optimizer: # perform learning
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_accuracy += accuracy
            running_loss += loss.item()

        epoch_accuracy, epoch_loss = running_accuracy/len(dataloader), running_loss/len(dataloader)
        
        if not optimizer: # without learning, function execution ends here ------------------------------------
            print("Final model test accuracy: {}.".format(epoch_accuracy))
            return predictions, epoch_accuracy, epoch_loss
        
        _, epoch_validation_accuracy, epoch_validation_loss = eval_model(model, loss_func, validation_dataloader)

        loss_arr.append(epoch_loss)
        accuracy_arr.append(epoch_accuracy)
        validation_loss_arr.append(epoch_validation_loss)
        validation_accuracy_arr.append(epoch_validation_accuracy)

        print("    EPOCH {}: train acc {}, val acc {} || train loss {}, val loss {}.".format(
            epoch+1, accuracy_arr[-1], validation_accuracy_arr[-2], loss_arr[-1], validation_loss_arr[-2]))
    
    write_run_plots(writer, loss_arr, accuracy_arr, validation_accuracy_arr, validation_loss_arr)
    writer.close()
    print("Training completed in {} seconds.".format(time.time() - start_time))


def fit_model(writer, run_spec, model, train_dl, loss_func, optimizer, valid_dl, verbose):
    '''Trains the model. For each epoch, iterates through all batches, 
    calculating loss and optimizing model weights after each batch. The 
    running loss is also saved to tensorboard. 
    '''
    print('BEGIN TRAINING: {} model with a \'{}\' optimization and \'{}\' augmentation over {} epochs'.format(
        run_spec["model_str"], run_spec["optimizer"], run_spec["augmentation"], run_spec["epochs"]))


    start_time = time.time()

    # Validation accuracy V_e for epoch e is defined as the accuracy the model achives over the validation set 
    # after completing e-1 epochs of training. The loss L_e and train accuracy T_e for epoch e are defined as
    # the averaged loss and accuracy over all training batches in epoch e. Eg. the validation accuracy of epoch 
    # 1 is 0, and all future valued are offset by one. Without this shift, the model that validation accuracy is
    # calculated over would have undergone more training, artificially inflatin its value. 
    validation_accuracy_arr, validation_loss_arr, train_accuracy_arr, train_loss_arr = [0], ["NA"], [], []

    for epoch in range(run_spec["epochs"]):
        running_loss, running_train_accuracy = 0.0, 0.0

        for i, (x_batch, y_batch) in enumerate(train_dl):

            # print only once, across all batches and epochs
            label_str = "batch" if (verbose and not i and not epoch) else None  
            yh_batch, predictions, accuracy, loss = model_wrapper(label_str, model, loss_func, x_batch, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_train_accuracy += accuracy

        epoch_train_loss, epoch_train_accuracy = running_loss/len(train_dl), running_train_accuracy/len(train_dl)
        epoch_validation_accuracy, epoch_validation_loss = eval_model(None, valid_dl, model, loss_func, run_spec)

        train_loss_arr.append(epoch_train_loss)
        train_accuracy_arr.append(epoch_train_accuracy)
        validation_loss_arr.append(epoch_validation_loss)
        validation_accuracy_arr.append(epoch_validation_accuracy)

        print("    EPOCH {}: train acc {}, val acc {} || train loss {}, val loss {}.".format(
            epoch+1, train_accuracy_arr[-1], validation_accuracy_arr[-2], train_loss_arr[-1], validation_loss_arr[-2]))
    
    write_run_plots(writer, train_loss_arr, train_accuracy_arr, validation_accuracy_arr, validation_loss_arr)
    writer.close()
    print("Training completed in {} seconds.".format(time.time() - start_time))


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
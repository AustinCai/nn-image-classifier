import torch
from torch import nn 
import torch.nn.functional as F
import time

import visualize


#model
class Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, xb):
        return self.lin(xb)


class SmallNN(nn.Module):
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

class LargeNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        IN_HIDDEN, OUT_HIDDEN = 256, 256
        self.l1 = nn.Linear(in_channels, IN_HIDDEN)
        self.l2 = nn.Linear(IN_HIDDEN, OUT_HIDDEN)
        self.l3 = nn.Linear(IN_HIDDEN, OUT_HIDDEN)
        self.l4 = nn.Linear(IN_HIDDEN, OUT_HIDDEN)
        self.l5 = nn.Linear(IN_HIDDEN, OUT_HIDDEN)
        self.l6 = nn.Linear(OUT_HIDDEN, out_channels) 

    def forward(self, xb):
        a1 = F.relu(self.l1(xb))
        a2 = F.relu(self.l2(a1))
        a3 = F.relu(self.l3(a2))
        a4 = F.relu(self.l4(a3))
        a5 = F.relu(self.l5(a4))
        return self.l6(a5)

class SmallCNN(nn.Module):
    out_conv = 18
    in_hidden, out_hidden = 64, 64
    cifar10_x, cifar10_y = 32, 32
    stride_pool, out_pool = 2, None

    # TODO - get rid of in_channels
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_pool = int(self.out_conv * (self.cifar10_x/self.stride_pool) * (self.cifar10_y/self.stride_pool))

        self.conv1 = nn.Conv2d(3, self.out_conv, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=self.stride_pool, padding=0)
        self.conv2 = nn.Conv2d(self.out_conv, self.out_conv, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(int(self.out_pool / self.stride_pool**2), self.out_hidden)
        self.fc2 = nn.Linear(self.in_hidden, out_channels)

    def forward(self, x):
        a1 = F.relu(self.conv1(x))
        a1 = self.pool(a1)
        a2 = F.relu(self.conv2(a1))
        a2 = self.pool(a2)
        a3 = a2.view(-1, int(self.out_pool / self.stride_pool**2))
        a3 = F.relu(self.fc1(a3))
        return self.fc2(a3)

# https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
class BestCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # corresponds to model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.do1 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.do2 = nn.Dropout(0.3)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.do3 = nn.Dropout(0.4)

        self.fc1 = nn.Linear(8192, out_channels)

    # TODO - add batch norm, dropout
    def forward(self, x):
        a1 = F.relu(self.conv1(x))
        a1 = self.bn1(a1)
        a2 = F.relu(self.conv2(a1))
        a2 = self.bn1(a2)
        a3 = self.pool(a2)
        a3 = self.do1(a3)

        a4 = F.relu(self.conv3(a3))
        a4 - self.bn2(a4)
        a5 = F.relu(self.conv4(a4))
        a5 = self.bn2(a5)
        a6 = self.pool(a5)
        a6 = self.do2(a6)

        a7 = F.relu(self.conv5(a6))
        a7 = self.bn3(a7)
        a8 = F.relu(self.conv6(a7))
        a8 = self.bn3(a8)
        a8 = self.do3(a8)

        a9 = a8.view(-1, 8192)
        return self.fc1(a9)


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
            if not i and not epoch: 
                print("    x_batch.size(): {}".format(x_batch.size()))
            yh_batch = model(x_batch)
            if not i and not epoch: 
                print("    yh_batch.size(): {}".format(yh_batch.size()))

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
                    if not epoch:
                        print("    x_valid.size(): {}".format(x_valid.size()))
                    yh_valid = model(x_valid)   
                    if not epoch:
                        print("    yh_valid.size(): {}".format(yh_valid.size()))

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


def init_optimizer(run_spec, model):
    if run_spec["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=run_spec["lr"])
    if run_spec["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=run_spec["lr"])
    raise Exception("Invalid optimizer specification of {}.".format(run_spec["optimizer"]))


def init_model(model_type, dataset, dev):
    MNIST_IN_CHANNELS, CIFAR10_IN_CHANNELS = 784, 3072
    in_channels = MNIST_IN_CHANNELS if dataset == "mnist" else CIFAR10_IN_CHANNELS
    out_channels = 10

    if model_type == "small_cnn":
        return SmallCNN(in_channels, out_channels).to(dev)
    if model_type == "large_nn":
        return LargeNN(in_channels, out_channels).to(dev)
    if model_type == "small_nn": 
        return SmallNN(in_channels, out_channels).to(dev)
    if model_type == "linear": 
        return Linear(in_channels, out_channels).to(dev)
    if model_type == "best_cnn":
        return BestCNN(in_channels, out_channels).to(dev)
    raise Exception("Invalid model_str specification of {}.".format(model_type))
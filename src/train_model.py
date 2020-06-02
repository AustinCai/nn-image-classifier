import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import argparse
import sys
from pathlib import Path

import util 
import data_loading
import visualize
import training
import models

import pickle
from PIL import Image
import torchvision.transforms as tfs
from pathlib import Path
import progressbar

from util import Constants
from util import Objects

def get_args(arguments):
    '''Parse the arguments passed via the command line.
    '''
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', 
        help='Print debugging output', action='store_true')
    parser.add_argument('-e', '--epochs', 
        help='Epochs to run training', type=int, default=10)
    parser.add_argument('-m', '--load_model', 
        help = 'Load a specific model to continue training.', type=str)
    parser.add_argument('-a', '--augmentation', 
        help = 'Specify augmentation to use.', type=str, default="none")
    parser.add_argument('-d', '--dataset',
        help = 'Specify which dataset to train over. If the dataset is a built-in dataset, only'
        + 'the dataset name (eg. cifar10) must be specified. Otherwise, the entire path must be' 
        + 'specified (eg. saved_data/gmaxup_cifar-10-batches).', 
        type=str, default="cifar10")
    parser.add_argument('-s', '--save_model', 
        help = 'Save model after training.', action='store_true')
    parser.add_argument('-f', '--fast', 
        help = 'Fast testing mode, does not properly train model.', action='store_true')
    parser.add_argument('-o', '--optimizer', 
        help = 'Specify which optimizer to use. Default adam.', 
        type=str, default="adam")
    parser.add_argument('-n', '--name', 
        help = 'Add optional string to save files.', 
        type=str)
    args = parser.parse_args(arguments)
    return args


def build_save_str(args):
    optional_tokens = [] 
    if args.load_model:
        optional_tokens.append("pretrained_{}e".format(pretrained_epochs))
    if args.name:
        optional_tokens.append(args.name)
    optional_str = ""
    if len(optional_tokens):
        for token in optional_tokens:
            optional_str += "{}-".format(token)

    return '{}-{}e-{}-{}{}'.format(
        Constants.model_str, 
        args.epochs, 
        "gmaxup" if "gmaxup_cifar" in args.dataset else args.augmentation, # augmentation string
        optional_str, # optional string 
        datetime.datetime.now().strftime("%-m.%d.%y-%H:%M"))


def main(args=None):
    '''Iterates through each model configuration specified by run_specifications. 
    Initializes and trains each model with its specified configuration, 
    evaluates each on the test set, and records its performance to 
    tensorboard.
    '''
    print("Using GPU: {}".format(torch.cuda.is_available()))

    model = getattr(training.Models, Constants.model_str)
    optimizer = training.init_optimizer(model, args.optimizer)
    loss_func = torch.nn.CrossEntropyLoss() 

    # handle model loading, if the flag is set 
    if args.load_model == "wide-resnet":
        model = torchvision.models.wide_resnet50_2(pretrained=False, progress=True)
    elif args.load_model:
        model, optimizer, pretrained_epochs, loss = training.load_model(args.load_model, model, optimizer)
        print("Loaded model with {} epochs and {} loss.".format(pretrained_epochs, loss))

    Constants.save_str = build_save_str(args)
    writer = SummaryWriter(Path(__file__).parent.parent / "runs" / Constants.save_str)

    train_dlr, valid_dlr, test_dlr = data_loading.build_wrapped_dl(
        args.augmentation, args.dataset, verbose=args.verbose)

    bar = progressbar.ProgressBar(max_value=args.epochs*len(train_dlr)*Constants.batch_size, max_error=False)

    # show images and model graph 
    images, _ = iter(train_dlr).__next__()
    visualize.show_images(writer, images, Constants.batch_size, title="Images", verbose=args.verbose)
    visualize.show_graph(writer, model, images)

    print('Training {} model with a \'{}\' optimization and \'{}\' augmentation over {} epochs'.format(
        Constants.model_str, args.optimizer, args.augmentation, args.epochs), 
        file = open(Path(__file__).parent.parent / "logs" / '{}.txt'.format(Constants.save_str), 'a'))

    epoch = None
    train_acc, train_loss = None, None
    validation_acc, validation_loss = None, None

    try: # run, log training 
        for epoch in range(args.epochs):

            with torch.no_grad(): # validation
                validation_acc, validation_loss = training.run_epoch(
                    model, loss_func, valid_dlr, verbose=args.verbose, fast=args.fast)

            # training
            train_acc, train_loss = training.run_epoch(
                model, loss_func, train_dlr, epoch, bar, optimizer, args.verbose, args.fast)

            visualize.write_epoch_stats(
                writer, epoch, validation_acc, validation_loss, train_acc, train_loss)

    except: # gracefully handle crashes during training
        training.save_model(model, optimizer, train_loss, "CRASH-{}".format(Constants.save_str), epoch)
        raise Exception('Error on epoch {} of training. Data dumped.'.format(epoch))

    if args.save_model:
        training.save_model(model, optimizer, train_loss, Constants.save_str, args.epochs)

    # get test accuracy 
    test_acc, _ = training.run_epoch(model, loss_func, test_dlr, verbose=args.verbose, fast=args.fast)
    visualize.print_final_model_stats(train_acc, validation_acc, test_acc)

    writer.close()


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    print("args: {}".format(args))
    main(args=args)


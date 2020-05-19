import matplotlib
matplotlib.use('tkagg')  # Or any other X11 back-end
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import argparse
import sys
from pathlib import Path
import torchvision.transforms as tfs

import util 
import data_loading
import visualize
import training
import augmentations

from util import Constants
from util import Objects
import util

import pickle
import gzip
from PIL import Image
from collections import OrderedDict


def get_args(arguments):
    '''Parse the arguments passed via the command line.
    '''
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', help='Print debugging output', action='store_true')
    parser.add_argument('-l', '--load_model', help = 'Load a specific model', type=str)
    parser.add_argument('-f', '--fast', help = 'Fast testing mode, does not properly train model.', action='store_true')
    args = parser.parse_args(arguments)
    return args


def print_augment_stats(augment_stats):
    for augment_str, stats in augment_stats.items():
        print("{}:".format(augment_str))
        print("    incorrect/count: {}/{}".format(stats["incorrect_preds"], stats["count"]))
        print("    average loss: {}".format(stats["average_loss"]))


def run_gmaxup_on_sample(x, y, loss_func, model, augmented_batch, augment_stats=None):
    x = np.transpose(np.reshape(x, (3, 32, 32)), (1, 2, 0))
    x = Image.fromarray(x)

    max_loss = -1
    max_xy_tuple = None
    max_augment = None
    max_pred_correct = None # whether the loss maximizing augmentation resulted in a correct classification

    for op_num, ops in enumerate(augmentations.augment_list()):

        (op, minval, maxval) = ops
        val = (Constants.randaugment_m / 30) * float(maxval - minval) + minval

        x_aug_img = op(x, val)
        # first dimension = 1 to simulate a batch size of 1
        x_aug_tensor = Objects.transform_pil_image_to_tensor(x_aug_img).view(1, 3, 32, 32).to(Objects.dev)

        y = torch.tensor([y]).to(Objects.dev)
        yh, predictions = training.model_wrapper(model, x_aug_tensor)
        loss = loss_func(yh, y).item()

        pred_correct = predictions.item() == y.item

        if loss > max_loss:
            max_loss = loss
            max_xy_tuple = (x_aug_tensor.squeeze().cpu(), y.item()) # squeeze() to undo the first dimension=1
            max_augment = augmentations.augment_list_str()[op_num]

    augmented_batch.append(max_xy_tuple)

    if augment_stats:
        augment_stats[max_augment]["count"] += 1
        augment_stats[max_augment]["average_loss"] += max_loss
        if not pred_correct:
            augment_stats[max_augment]["incorrect_preds"] += 1 


def main(args):

    model_str = args.load_model if args.load_model else "saved_models/best_cnn-10e-0.001lr-none-adam-15:45:51"
    model, optimizer, epoch, loss = training.load_model(model_str)
    loss_func = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter(Path(__file__).parent.resolve() / '../runs/gen_maxup')

    dataset_str = "cifar-10-batches-py"
    batches_to_augment = \
        ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

    augmented_batch = []
    augment_stats = {}
    for augment_str in augmentations.augment_list_str():
        augment_stats[augment_str] = {"count": 0, "average_loss": 0.0, "incorrect_preds": 0}

    for curr_batch_str in batches_to_augment:
        images, labels = data_loading.load_data_by_path(
            Path("data") / '{}/{}'.format(dataset_str, curr_batch_str))        

        for sample_num, (x, y) in enumerate(zip(images, labels)):

            if not sample_num % 100:
                print(sample_num)
            if args.fast and sample_num > 200:
                break

            run_gmaxup_on_sample(x, y, loss_func, model, augmented_batch, augment_stats)

        for key in augment_stats:
            if augment_stats[key]["count"] > 0:
                augment_stats[key]["average_loss"] /= augment_stats[key]["count"]

    augmented_cifar10_ds = data_loading.DatasetFromTupleList(augmented_batch)
    print(augmented_cifar10_ds[0])
    with open(Path("gmaxup_data") / "gmaxup-data-batch", 'wb') as handle:
        pickle.dump(augmented_cifar10_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print_augment_stats(augment_stats)

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    print("args: {}".format(args))
    main(args)
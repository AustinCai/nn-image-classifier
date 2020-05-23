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
import random

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
import heapq

def get_args(arguments):
    '''Parse the arguments passed via the command line.
    '''
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', help='Print debugging output', action='store_true')
    parser.add_argument('-l', '--load_model', help = 'Load a specific model', type=str)
    parser.add_argument('-f', '--fast', help = 'Fast testing mode, does not properly train model.', action='store_true')
    parser.add_argument('-c', '--composition', help = 'Number of augmentations in a composition.', type=int, default=2) #L in the paper
    parser.add_argument('-m', '--multiplier', help = 'Number of augmented data points per input.', type=int, default=1) #S in the paper
    args = parser.parse_args(arguments)
    return args


def print_augment_stats(augment_stats):
    for augment_str, stats in augment_stats.items():
        print("{}:".format(augment_str))
        print("    incorrect/count: {}/{}".format(stats["incorrect_preds"], stats["count"]))
        print("    average loss: {}".format(stats["average_loss"]))


def compose_transormations(op_idxs, x_img):
    used_augments = []

    for op_idx in op_idxs:
        used_augments.append(augmentations.augment_list_str()[op_idx])
        op, minval, maxval = augmentations.augment_list()[op_idx]
        val = (Constants.randaugment_m / 30) * float(maxval - minval) + minval
        x_img = op(x_img, val)

    return x_img, used_augments


def run_gmaxup_on_sample(sample_num, x, y, loss_func, model, augmented_batch, args, augment_stats=None):
    x_img = Image.fromarray(np.transpose(np.reshape(x, (3, 32, 32)), (1, 2, 0)))

    xy_tuple_list_by_loss = []
    heapq.heapify(xy_tuple_list_by_loss)

    for c in range(4): # number of compositions to sample, C
        op_idxs = random.sample(range(16), args.composition)
        x_aug_img, used_augments = compose_transormations(op_idxs, x_img)

        # first dimension = 1 to simulate a batch size of 1
        x_aug_tensor = Objects.transform_pil_image_to_tensor(x_aug_img).view(1, 3, 32, 32).to(Objects.dev)

        y = torch.tensor([y]).to(Objects.dev)
        yh, predictions = training.model_wrapper(model, x_aug_tensor)
        loss = loss_func(yh, y).item()

        max_xy_tuple = (x_aug_tensor.squeeze().cpu(), y.item()) # squeeze() to undo the first dimension=1
        
        # sample_num+c is a unique integer to arbitrarily break ties
        heapq.heappush(xy_tuple_list_by_loss, (loss, sample_num+c, used_augments, max_xy_tuple))

    for loss, _, used_augments, xy_tuple in heapq.nlargest(args.multiplier, xy_tuple_list_by_loss):
        augmented_batch.append(xy_tuple)

        if augment_stats:
            for augment_str in used_augments:
                augment_stats[augment_str]["count"] += 1
                augment_stats[augment_str]["average_loss"] += loss


def main(args):

    model_str = args.load_model if args.load_model else "saved_models/best_cnn-10e-0.001lr-none-adam-15:45:51"
    model = training.init_model()
    optimizer = training.init_optimizer(model)
    model, optimizer, epoch, loss = training.load_model(model_str, model, optimizer)
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
            if args.fast and sample_num > 50:
                break

            run_gmaxup_on_sample(sample_num, x, y, loss_func, model, augmented_batch, args, augment_stats)

        for key in augment_stats:
            if augment_stats[key]["count"] > 0:
                augment_stats[key]["average_loss"] /= augment_stats[key]["count"]

    augmented_cifar10_ds = data_loading.DatasetFromTupleList(augmented_batch)
    print(augmented_cifar10_ds[0])
    with open(Path("gmaxup_data") / "gmaxup_cifar10-{}l-{}s".format(args.composition, args.multiplier), 'wb') as handle:
        pickle.dump(augmented_cifar10_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print_augment_stats(augment_stats)

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    print("args: {}".format(args))
    main(args)
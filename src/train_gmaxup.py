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
import build_model
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
        x_aug_tensor = Objects.transform_pil_image_to_tensor(x_aug_img).view(1, 3, 32, 32).to(Objects.dev)

        y = torch.tensor([y]).to(Objects.dev)
        yh, predictions = build_model.model_wrapper(model, x_aug_tensor)
        loss = loss_func(yh, y).item()

        pred_correct = predictions.item() == y.item

        if loss > max_loss:
            max_loss = loss
            max_xy_tuple = (x_aug_img, y.item())
            max_augment = augmentations.augment_list_str()[op_num]

    augmented_batch.append(max_xy_tuple)

    if augment_stats:
        augment_stats[max_augment]["count"] += 1
        augment_stats[max_augment]["average_loss"] += max_loss
        if not pred_correct:
            augment_stats[max_augment]["incorrect_preds"] += 1 


def main(args):
    if not args.load_model:
        raise Exception("Must specify a trained model to load.")

    model, optimizer, epoch, loss = build_model.load_model(args.load_model)
    loss_func = torch.nn.CrossEntropyLoss()

    # pil_to_tensor = tfs.Compose([tfs.ToTensor(), tfs.Normalize((0.5,), (0.5,))]) # redundant with baseline_transforms
    writer = SummaryWriter(Path(__file__).parent.resolve() / '../runs/gen_maxup')

    images, labels = data_loading.load_data_by_path(Path("data") / 'cifar-10-batches-py/data_batch_1')

    # list of (x, y) tuples
    augmented_batch = []
    augment_stats = {}
    for augment_str in augmentations.augment_list_str():
        augment_stats[augment_str] = {"count": 0, "average_loss": 0.0, "incorrect_preds": 0}

    for sample_num, (x, y) in enumerate(zip(images, labels)):

        if not sample_num % 100:
            print(sample_num)
        if sample_num > 100:
            break

        run_gmaxup_on_sample(x, y, loss_func, model, augmented_batch, augment_stats)


    for key in augment_stats:
        if augment_stats[key]["count"] > 0:
            augment_stats[key]["average_loss"] /= augment_stats[key]["count"]

    print("DONE: len(augmented_batch): {}".format(len(augmented_batch)))
    print_augment_stats(augment_stats)

    data = {"augmented_batch": augmented_batch, "stats": augment_stats}
    identifier = "len{}-{}".format(len(augmented_batch), datetime.datetime.now().strftime("%m.%d.%Y-%H:%M:%S"))
    util.pickle_save(data, Path("augmented_data"), identifier, "aug-batch")

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    print("args: {}".format(args))
    main(args)
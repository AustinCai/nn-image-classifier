# python src/train_gmaxup.py -m saved_models/best_cnn-200e-none-11:53-5.26.20 -r reduced2

import matplotlib
matplotlib.use('tkagg')  # Or any other X11 back-end

import torch
import numpy as np
import argparse
import sys
import random
import pickle
import heapq
import progressbar

import data_loading
import display
import training
import augmentations
import util

from pathlib import Path
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from util import Constants, Objects, BasicTransforms, GMaxupConsts


def get_args(arguments):
    '''Parse the arguments passed via the command line.
    '''
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', 
        help='Print debugging output', action='store_true')
    parser.add_argument('-m', '--load_model', # format saved_models/XXXXX
        help = 'Load a specific model', type=str)
    parser.add_argument('-f', '--fast', 
        help = 'Fast testing mode, does not properly train model.', type=int)
    parser.add_argument('-t', '--test',
        help = 'Testing mode. Outputs images, only processes one sample.', action='store_true')
    parser.add_argument('-l', '--layers', 
        help = 'Number of augmentations to layer.', type=int, default=2) #L in the paper
    parser.add_argument('-c', '--choices', 
        help = 'Number of augmented compositions to sample and choose from.', type=int, default=4)
    parser.add_argument('-s', '--setsize', 
        help = 'Number of augmented data points per input.', 
        type=int, default=1) #S in the paper
    parser.add_argument('-r', '--range',
        help = 'Range setting for transformation magnitudes.',
        type=str, default='orig')
    parser.add_argument('-d', '--dataset', 
        help = 'Specify which dataset to augment. Default cifar-10-batches-py.', 
        type=str, default="cifar-10-batches-py")
    parser.add_argument('-n', '--name', 
        help = 'Dataset name.', 
        type=str)
    args = parser.parse_args(arguments)
    return args


def apply_transformations(op_idxs, x_img, magnitude_range):
    '''Applies the transformations referenced in op_idxs to a pillow image.'''

    used_augment_mag_tuples = [] # to populate augment_stats

    for op_idx in op_idxs:
        op, minval, maxval, op_str = augmentations.augment_list(magnitude_range)[op_idx]
        
        magnitude = GMaxupConsts.trans_magnitude
        # magnitude = random.randint(0, 30)
        val = (magnitude / 30) * float(maxval - minval) + minval
        used_augment_mag_tuples.append((op_str, magnitude))
        x_img = op(x_img, val)

    return x_img, used_augment_mag_tuples


def run_gmaxup_on_sample(sample_num, x, y, loss_func, model, augmented_batch, args, 
    augment_stats=None, writer=None):
    x_img = Image.fromarray(np.transpose(np.reshape(x, (3, 32, 32)), (1, 2, 0)))
    y = torch.tensor(y).view(1).to(Objects.dev)

    xy_tuple_list_by_loss = []
    heapq.heapify(xy_tuple_list_by_loss)

    for c in range(args.choices): # number of layers to sample, C
        operation_idxs = random.sample(range(len(augmentations.augment_list(args.range))), args.layers)
        x_aug_img, used_augment_mag_tuples = apply_transformations(operation_idxs, x_img, args.range) 

        # first dimension = 1 to simulate a batch size of 1
        x_aug_tensor = BasicTransforms.baseline(x_aug_img).view(1, 3, 32, 32).to(Objects.dev)
        
        yh, _ = training.model_wrapper(model, x_aug_tensor)
        loss = loss_func(yh, y).item()

        if writer and args.test: # show augmented candidates for one sample 
            display.log_image_and_transformations(writer, x_aug_tensor, c, used_augment_mag_tuples, loss, sample_num)

        x_out = x_aug_tensor.squeeze().cpu()
        y_out = y.item()
        xy_tuple = (x_out, y_out) # squeeze() to undo the first dimension=1
        
        # sample_num+c is a unique integer to arbitrarily break ties
        heapq.heappush(xy_tuple_list_by_loss, (loss, sample_num+c, used_augment_mag_tuples, xy_tuple))

    for loss, img_id, used_augment_mag_tuples, xy_tuple in heapq.nlargest(args.setsize, xy_tuple_list_by_loss):
        augmented_batch.append(xy_tuple)

        if writer and args.test:
            print("chosen img_id: {}".format(img_id))
            display.show_images(
                writer, xy_tuple[0].view(1, 3, 32, 32), 1, title="Chosen_Image".format(c))

        if augment_stats:
            for (augment_str, magnitude) in used_augment_mag_tuples:
                augment_stats[augment_str]["count"] += 1
                augment_stats[augment_str]["average_mag"] += magnitude
                augment_stats[augment_str]["average_loss"] += loss


def main(args):

    if args.test:
        print("Testing mode.")
        args.fast = 1
        args.choices = 20
        args.setsize = 1
    
    if not args.load_model:
        saved_models_path = Path(__file__).parent.parent.resolve().parent / 'saved_models'
        model_path =  saved_models_path / saved_models_path.iterdir().__next__().name
    else:
        model_path = Path(__file__).parent.parent.resolve().parent / args.load_model
    print("Using evaluator model: {}".format(model_path))

    # load evaluator model
    model = getattr(training.Models, Constants.model_str)
    model, _, epoch, loss = training.load_model(model_path, model)
    loss_func = torch.nn.CrossEntropyLoss()

    GMaxupConsts.save_str = display.build_gmaxup_save_str(args)
    writer = SummaryWriter(Path(__file__).parent / '../runs/{}'.format(GMaxupConsts.save_str))

    # dictionary containing statistics on which transformations gmaxup selects
    augment_stats = {} 
    for _, _, _, augment_str in augmentations.augment_list(args.range):
        augment_stats[augment_str] = {"count": 0, "average_loss": 0.0, "incorrect_preds": 0, "average_mag": 0}
    
    augmented_batch = []

    # initializes loading bar
    dataset_batches = GMaxupConsts.dataset_batches_dict[args.dataset]
    samples_per_batch = args.fast if args.fast else GMaxupConsts.samples_per_batch
    bar = progressbar.ProgressBar(max_value=len(dataset_batches)*samples_per_batch, max_error = False)

    # iterates through all batches of the dataset 
    for batch_num, curr_batch_str in enumerate(dataset_batches):
        images, labels = data_loading.load_data_by_path(
            Path(__file__).parent.parent.resolve().parent / 'saved_data' / args.dataset / curr_batch_str)       

        # iterates through all samples in a batch
        for sample_num, (x, y) in enumerate(zip(images, labels)):
            if sample_num < samples_per_batch: # early exit if -f
                run_gmaxup_on_sample(
                    sample_num, x, y, loss_func, model, augmented_batch, args, augment_stats, writer)
                bar.update(batch_num*samples_per_batch + sample_num)

        for key in augment_stats:
            if augment_stats[key]["count"] > 0:
                augment_stats[key]["average_loss"] /= augment_stats[key]["count"]
                augment_stats[key]["average_mag"] /= augment_stats[key]["count"]

        if args.test:
            break

    augmented_cifar10_ds = data_loading.DatasetFromTupleList(augmented_batch)
    with open(Path(__file__).parent.parent / "gmaxup_data" / GMaxupConsts.save_str, 'wb') as handle:
        pickle.dump(augmented_cifar10_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    display.print_augment_stats(augment_stats)


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    print("args: {}".format(args))
    main(args)
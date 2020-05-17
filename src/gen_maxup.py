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
import build_model

from util import Constants

def get_args(arguments):
    '''Parse the arguments passed via the command line.
    '''
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', help='Print debugging output', action='store_true')
    # parser.add_argument('-e', '--epochs', help='Epochs to run training', type=int, default=10)
    parser.add_argument('-l', '--load_model', help = 'Load a specific model', type=str)
    # parser.add_argument('-f', '--fast', help = 'Fast testing mode, does not properly train model.', action='store_true')

    args = parser.parse_args(arguments)
    return args

def main(run_spec):

    model = build_model.init_model(run_spec["model_str"])
	if args.load_model:
        model.load_state_dict(torch.load(
            Path(__file__).parent.resolve() / '../{}'.format(args.load_model)))
        model.eval() # sets dropout and batch normalization layers

    baseline_transforms = data_loading.init_baseline_transforms()
    writer = SummaryWriter(Path(__file__).parent.resolve() / '../runs/gen_maxup')


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    print("args: {}".format(args))
	main({
            "model_str": "best_cnn", 
            "epochs": 50, 
            "lr": 1e-3, 
            "augmentation": "random",
            "batch_size": 64, 
            "optimizer": "adam"
        })
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

#maxup
def main(run_spec):

	print("Using GPU: {}".format(torch.cuda.is_available()))
    baseline_transforms = data_loading.init_baseline_transforms()
    writer = SummaryWriter(
        Path(__file__).parent.resolve() / '../runs/gen-maxup-{}'.format(
        datetime.datetime.now().strftime("%H:%M:%S")))

    train_dl, valid_dl, test_dl = data_loading.build_dl("none", baseline_transforms, args.verbose)
    reshape = False if "cnn" in run_spec["model_str"] else True
    train_dlr, valid_dlr, test_dlr = data_loading.wrap_dl(train_dl, valid_dl, test_dl, reshape, args.verbose)

    loss_func = torch.nn.CrossEntropyLoss() # TODO: hyperparameterize 
    model = build_model.init_model(run_spec["model_str"])
    optimizer = build_model.init_optimizer(run_spec, model)



if __name__ == "__main__":
	main({
            "model_str": "best_cnn", 
            "epochs": 50, 
            "lr": 1e-3, 
            "augmentation": "random",
            "batch_size": 64, 
            "optimizer": "adam"
        })
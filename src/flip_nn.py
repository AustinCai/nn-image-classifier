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


# TODO: new CNN that outputs image predictions
# TODO: fold into training process
def main(run_spec):
    print("Using GPU: {}".format(torch.cuda.is_available()))
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    baseline_transforms = data_loading.init_baseline_transforms()
    train_dl, valid_dl, test_dl = data_loading.build_dl(
        {"augmentation": "none", "batch_size": 64}, "cifar10", baseline_transforms)
    hflip_train_dl, hflip_valid_dl, hflip_test_dl = data_loading.build_dl(
        {"augmentation": "hflip_all", "batch_size": 64}, "cifar10", baseline_transforms)

    reshape = False if "cnn" in run_spec["model_str"] else True
    raw_dl, train_dlr, valid_dlr, test_dlr = data_loading.wrap_dl(
        train_dl, valid_dl, test_dl, dev, reshape, 
        hflip_train_dl, hflip_valid_dl, hflip_test_dl)

    loss_func = torch.nn.CrossEntropyLoss() # TODO: hyperparameterize 
    model = build_model.init_model(run_spec["model_str"], dataset, dev)
    optimizer = build_model.init_optimizer(run_spec, model)

    images, hflip_images = iter(raw_dl).__next__()
    visualize.show_images(writer, images, title="original")
    visualize.show_images(writer, hflip_images, title="flipped")

    visualize.show_graph(writer, model, images, run_spec["batch_size"])

if __name__ == "__main__":
    main({
            "model_str": "large_nn", 
            "epochs": 10, 
            "lr": 1e-3, 
            "augmentation": "none", 
            "batch_size": 64, 
            "optimizer": "adam"
        })
    print("done")
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
    dataset = "cifar10"

    writer = SummaryWriter(
        Path(__file__).parent.resolve() / '../runs/flip-nn-test-{}'.format(
        datetime.datetime.now().strftime("%H:%M:%S")))

    print("Using GPU: {}".format(torch.cuda.is_available()))
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    baseline_transforms = data_loading.init_baseline_transforms()
    train_dl, valid_dl, test_dl = data_loading.build_dl(
        {"augmentation": "none"}, dataset, baseline_transforms, shuffle=False)
    hflip_train_dl, hflip_valid_dl, hflip_test_dl = data_loading.build_dl(
        {"augmentation": "hflip_all"}, dataset, baseline_transforms, shuffle=False)

    reshape = False if "cnn" in run_spec["model_str"] else True
    print("before wrap_dl()")
    train_dlr, valid_dlr, test_dlr = data_loading.wrap_dl(
        train_dl, valid_dl, test_dl, dev, reshape, 
        hflip_train_dl, hflip_valid_dl, hflip_test_dl, verbose=True)

    # loss_func = torch.nn.CrossEntropyLoss() # TODO: hyperparameterize 
    loss_func = torch.nn.L1Loss()
    model = build_model.init_model(run_spec["model_str"], dataset, dev)
    optimizer = build_model.init_optimizer(run_spec, model)

    images, hflip_images = iter(train_dlr).__next__()
    visualize.show_images(writer, images, title="original_2", verbose=True)
    visualize.show_images(writer, hflip_images, title="flipped_2", verbose=True)
    visualize.show_graph(writer, model, images)

    last_epoch_stats = build_model.run_training(model, loss_func, train_dlr, valid_dlr, optimizer, writer, run_spec, verbose=True)[-1]
    accuracy, loss = build_model.run_epoch(model, loss_func, test_dlr, verbose=True)

    print("Final model statistics:")
    print("    training accuracy: {}".format(last_epoch_stats["accuracy"]))
    print("    validation accuracy: {}".format(last_epoch_stats["validation_accuracy"]))
    print("    train/val difference: {}".format(
        last_epoch_stats["accuracy"] - last_epoch_stats["validation_accuracy"]))
    print("    test accuracy: {}".format(accuracy))


    writer.add_hparams(run_spec, {'accuracy': accuracy})
    writer.close()
    print("========================= End of Run =========================\n")


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
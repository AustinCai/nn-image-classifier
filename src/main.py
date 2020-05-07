import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import argparse
import sys

import util 
import data_loading
import visualize
import build_model


torch.manual_seed(42)
np.random.seed(42)


def get_args(arguments):
    '''Parse the arguments passed via the command line.
    '''
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', help='Print debugging output, with 0 being no output', action='count', default=1)
    parser.add_argument('-e', '--epochs', help='Epochs to run training', type=int, default=3)
    args = parser.parse_args(arguments)
    return args


# TODO:
# - take command line params
# - plot test and validation accuracy together on tensorbaord
def main(run_specifications, train_batch_size=60, dataset="mnist", args=None):
    util.assert_params(run_specifications, train_batch_size, dataset)
    
    print("Using GPU: {}".format(torch.cuda.is_available()))
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    test_trans, vflip_trans, hflip_trans, contrast_trans, rand_trans = data_loading.init_transforms()
    transforms = {"none": test_trans, 
                  "vflip": vflip_trans,
                  "hflip": hflip_trans, 
                  "contrast": contrast_trans,
                  "random": rand_trans}

    for run_spec in run_specifications:
        '''Iterates through each model configuration specified by run_specifications. 
        Initializes and trains each model with its specified configuration, 
        evaluates each on the test set, and records its performance to 
        tensorboard.
        '''
        torch.manual_seed(42)
        np.random.seed(42)
        writer = SummaryWriter('runs/{}-{}e-{}lr-{}-{}bs-{}-{}'.format(
            run_spec["model_str"], run_spec["epochs"], run_spec["lr"], 
            run_spec["augmentation"], run_spec["batch_size"], run_spec["optimizer"],
            datetime.datetime.now().strftime("%H:%M:%S")))

        train_dl, valid_dl, test_dl = data_loading.build_dl(run_spec, dataset, transforms, args.verbose)
        reshape = False if "cnn" in run_spec["model_str"] else True
        raw_dl, train_dlr, valid_dlr, test_dlr = data_loading.wrap_dl(train_dl, valid_dl, test_dl, dev, reshape, args.verbose)

        loss_func = torch.nn.CrossEntropyLoss() # TODO: hyperparameterize 
        model = build_model.init_model(run_spec["model_str"], dataset, dev)
        optimizer = build_model.init_optimizer(run_spec, model)

        # slower when this is uncommented, and it sometimes doesn't work
        visualize.show_inputs(writer, raw_dl, model, run_spec["batch_size"], args.verbose)

        build_model.fit_model(writer, run_spec, model, train_dlr, loss_func, optimizer, valid_dlr)
        build_model.eval_model(writer, test_dlr, model, run_spec, args.verbose)
        writer.close()
        print("========================= End of Run =========================\n")


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    print("args: {}".format(args))
    main(run_specifications = [{"model_str": "best_cnn", "epochs": 50, "lr": 1e-3, "augmentation": "random", "batch_size": 64, "optimizer": "adam"},
                               {"model_str": "best_cnn", "epochs": 50, "lr": 1e-3, "augmentation": "hflip", "batch_size": 64, "optimizer": "adam"}],
         dataset="cifar10", args=args)


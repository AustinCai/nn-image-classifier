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

# FFNNs behave deterministically without having to seed across all files
# seeding across all files is not sufficient for CNNs and randomly augmented models to behave deterministically
# def seed_all(s):
#     torch.manual_seed(s)
#     np.random.seed(s)
#     visualize.seed(s)
#     data_loading.seed(s)
#     build_model.seed(s)

def get_args(arguments):
    '''Parse the arguments passed via the command line.
    '''
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', help='Print debugging output', action='count', default=0)
    parser.add_argument('-e', '--epochs', help='Epochs to run training', type=int, default=3)
    args = parser.parse_args(arguments)
    return args


def main(run_specifications, train_batch_size=60, dataset="mnist", args=None):
    '''Iterates through each model configuration specified by run_specifications. 
    Initializes and trains each model with its specified configuration, 
    evaluates each on the test set, and records its performance to 
    tensorboard.
    '''
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
        # Notes on deterministic randomness:
        # seeds here are necessary and sufficient for determinism across FFNN models run within a single execution and across executions
        # using CNNs or a random augmentation policy destroys determinism, both across models run within a single execution and across executions
        # random seeding across all src files did not change this behavior
        torch.manual_seed(42)
        np.random.seed(42)
        # seed_all(42)

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

        # takes 3 minutes to run, comment if not needed
        # visualize.show_inputs_and_graph(writer, raw_dl, model, run_spec["batch_size"], args.verbose)

        build_model.train_and_eval_model(model, loss_func, train_dlr, valid_dlr, optimizer, writer, run_spec, verbose=args.verbose)
        accuracy, loss = build_model.run_epoch(model, loss_func, test_dlr, verbose=args.verbose)
        print("Final model test accuracy: {}.".format(accuracy))
        writer.add_hparams(run_spec, {'accuracy': accuracy})
        writer.close()
        print("========================= End of Run =========================\n")


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    print("args: {}".format(args))
    main(run_specifications = [{"model_str": "best_cnn", "epochs": 3, "lr": 1e-3, "augmentation": "random", "batch_size": 64, "optimizer": "adam"}],
         dataset="cifar10", args=args)


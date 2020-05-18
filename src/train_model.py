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
import models

import pickle
import gzip
from PIL import Image
import torchvision.transforms as tfs
from pathlib import Path

from util import Constants
from util import Objects

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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', help='Print debugging output', action='store_true')
    parser.add_argument('-e', '--epochs', help='Epochs to run training', type=int, default=10)
    parser.add_argument('-l', '--load_model', help = 'Load a specific model to continue training.', type=str)
    parser.add_argument('-s', '--save_model', help = 'Save model after training.', action='store_true')
    parser.add_argument('-f', '--fast', help = 'Fast testing mode, does not properly train model.', action='store_true')

    args = parser.parse_args(arguments)
    return args


def main(run_specifications, args=None):
    '''Iterates through each model configuration specified by run_specifications. 
    Initializes and trains each model with its specified configuration, 
    evaluates each on the test set, and records its performance to 
    tensorboard.
    '''
    # util.assert_params(run_specifications, dataset)
    
    print("Using GPU: {}".format(torch.cuda.is_available()))
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    baseline_transforms = data_loading.init_baseline_transforms()

    for run_spec in run_specifications:
        # Notes on deterministic randomness:
        # seeds here are necessary and sufficient for determinism across FFNN models run within a single execution and across executions
        # using CNNs or a random augmentation policy destroys determinism, both across models run within a single execution and across executions
        # random seeding across all src files did not change this behavior
        torch.manual_seed(42)
        np.random.seed(42)
        # seed_all(42)

        writer = SummaryWriter(
            Path(__file__).parent.resolve() / '../runs/{}-{}e-{}lr-{}-{}-{}'.format(
            run_spec["model_str"], args.epochs, run_spec["lr"], 
            run_spec["augmentation"], run_spec["optimizer"],
            datetime.datetime.now().strftime("%H:%M:%S")))

        train_dl, valid_dl, test_dl = data_loading.build_dl(
            run_spec["augmentation"], baseline_transforms, args.verbose)
        reshape = False if "cnn" in run_spec["model_str"] else True
        train_dlr, valid_dlr, test_dlr = data_loading.wrap_dl(
            train_dl, valid_dl, test_dl, reshape, args.verbose)

        loss_func = torch.nn.CrossEntropyLoss() # TODO: hyperparameterize 
        
        model = build_model.init_model(run_spec["model_str"])
        if args.load_model:
            model.load_state_dict(torch.load(
                Path(__file__).parent.resolve() / '../{}'.format(args.load_model)))
            model.eval() # sets dropout and batch normalization layers

        optimizer = build_model.init_optimizer(run_spec["optimizer"], run_spec["lr"], model)

        if not args.fast:
            images, _ = iter(train_dlr).__next__()
            visualize.show_images(writer, images, title="Images", verbose=args.verbose)
            visualize.show_graph(writer, model, images)

        print('Training {} model with a \'{}\' optimization and \'{}\' augmentation over {} epochs'.format(
            run_spec["model_str"], run_spec["optimizer"], run_spec["augmentation"], args.epochs))
        training_statistics_arr = build_model.run_training(
            model, loss_func, train_dlr, valid_dlr, optimizer, args)
        
        visualize.write_training_statistics(writer, training_statistics_arr)

        accuracy, loss = build_model.run_epoch(model, loss_func, test_dlr, verbose=args.verbose)

        if args.save_model:
            build_model.save_model(model, optimizer, training_statistics_arr[-1]["loss"], run_spec, args)

            # save_path = Path(__file__).parent.resolve() / '../models/{}-{}e-{}lr-{}-{}-{}'.format(
            #     run_spec["model_str"], args.epochs, run_spec["lr"], 
            #     run_spec["augmentation"], run_spec["optimizer"],
            #     datetime.datetime.now().strftime("%H:%M:%S"))

            # torch.save({
            #         'epoch': args.epochs,
            #         'model_state_dict': model.state_dict(), 
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'loss': training_statistics_arr[-1]["loss"]
            #     }, save_path)

        last_epoch_stats = training_statistics_arr[-1]
        visualize.print_final_model_stats(last_epoch_stats, accuracy)

        writer.add_hparams(run_spec, {'accuracy': accuracy})
        writer.close()
        print("========================= End of Run =========================\n")


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    print("args: {}".format(args))
    main(run_specifications = [
            {
                "model_str": "best_cnn",  
                "lr": 1e-3, 
                "augmentation": "none",
                "optimizer": "adam"
            }
        ],
        args=args
        )


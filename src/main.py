import torch
from torch import nn 
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import time
import datetime
import random

import util 
import data_loading
import visualize
import build_model

torch.manual_seed(42)
np.random.seed(42)


def main(run_specifications, train_batch_size=60, verbose=False, dataset="mnist"):
    if not util.validate_params(run_specifications, train_batch_size, verbose, dataset):
        return -1 
    
    print("Using GPU: {}".format(torch.cuda.is_available()))
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    test_trans, vflip_trans, hflip_trans, contrast_trans = data_loading.init_transforms()
    transforms = {"none": test_trans, "vflip": vflip_trans,
                        "hflip": hflip_trans, "contrast": contrast_trans}

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

        train_dl, valid_dl, test_dl = data_loading.build_dl(run_spec, dataset, transforms, verbose)
        raw_dl, train_dlr, valid_dlr, test_dlr = data_loading.wrap_dl(train_dl, valid_dl, test_dl, dev, verbose)

        loss_func = nn.CrossEntropyLoss() # TODO: hyperparameterize 
        model = build_model.init_model(run_spec["model_str"], dataset, dev)

        if run_spec["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=run_spec["lr"]) # TODO: hyperparameterize
        elif run_spec["optimizer"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=run_spec["lr"])
        else:
            print("ERROR: invalid optimizer")
            return -1 

        visualize.show_inputs(writer, raw_dl, model, run_spec["batch_size"], verbose)

        build_model.fit_model(writer, run_spec, model, train_dlr, loss_func, optimizer, valid_dlr)
        build_model.eval_model(writer, test_dlr, model, run_spec, verbose)
        writer.close()
        print("========================= End of Run =========================\n")


main(run_specifications = [{"model_str": "nn", "epochs": 1, "lr": 1e-3, "augmentation": "hflip", "batch_size": 64, "optimizer": "adam"}],
     verbose=True, dataset="cifar10")


# Tensorboard stuff, originally from Jupyter:


# Reinstalls tensorflow (which was uninstalled to get add_embeddings to work),
# which is required for tensorboard. If the installation reccomends you restart
# runtime, ignore it. 
# get_ipython().system('pip install tensorflow')


# import torch.utils.tensorboard

# get_ipython().run_line_magic('reload_ext', 'tensorboard')
# get_ipython().run_line_magic('tensorboard', '--logdir=runs')


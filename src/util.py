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

#Util
def validate_params(run_specifications, train_batch_size, verbose, dataset):
    if dataset not in {"mnist", "cifar10"}:
        print("ERROR: invalid dataset")
        return False
    for i, run_spec in enumerate(run_specifications):
        if run_spec["model_str"] not in {"nn", "linear"}:
            print("ERROR: invalid model in specification {}".format(i+1))
            return False
        if run_spec["augmentation"] not in {"none", "vflip", "hflip", "contrast"}:
            print("ERROR: invalid augmentation in specification {}".format(i+1))
            return False
        if run_spec["optimizer"] not in {"sgd", "adam"}:
            print("ERROR: invalid optimizer in specification {}".format(i+1))
            return False
    return True
    

def print_vm_info():
    '''Prints GPU and RAM info of the connected Google Colab VM.''' 
    gpu_info = get_ipython().getoutput('nvidia-smi')
    gpu_info = '\n'.join(gpu_info)
    if gpu_info.find('failed') >= 0:
        print('Select the Runtime → "Change runtime type" menu to enable a GPU accelerator, and then re-execute this cell.')
    else:
        print(gpu_info)

    from psutil import virtual_memory
    ram_gb = virtual_memory().total / 1e9
    print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

    if ram_gb < 20:
        print('To enable a high-RAM runtime, select the Runtime → "Change runtime type"')
        print('menu, and then select High-RAM in the Runtime shape dropdown. Then, re-execute this cell.')
    else:
        print('You are using a high-RAM runtime!')
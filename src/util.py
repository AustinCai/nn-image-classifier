import sys
import torch
import pickle
import torchvision.transforms as transforms
from pathlib import Path

import augmentations

# helpers for train_model.py and train_gmaxup.py ========================================================
# =======================================================================================================

class Objects:
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Constants:
    save_str = "untitled_save_str"

    cifar10_dim = (32, 32, 3)

    batch_size = 128
    learning_rate = 1e-3
    out_channels = 10
    model_str = "best_cnn"

    randaugment_n = 3
    randaugment_m = 4

    dataset_str = "cifar10"

class BasicTransforms:
    _CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    
    pil_image_to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)]
        )
    baseline = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            + [transforms.ToTensor(), transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)] 
        )
    random = transforms.Compose(
            [augmentations.RandAugment(Constants.randaugment_n, Constants.randaugment_m)]
            + [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            + [transforms.ToTensor(), transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)]
        )

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


def pickle_load(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)

# helpers for train_gmaxup.py ===========================================================================
# =======================================================================================================

def pickle_save(data, folder_path, identifier, data_str):
    with open(folder_path / "{}-{}".format(data_str, identifier), 'wb') as data_file:
        pickle.dump(data, data_file, protocol=pickle.HIGHEST_PROTOCOL)


# helpers for train_gmaxup.py ===========================================================================
# =======================================================================================================


# helpers for testing ===================================================================================
# =======================================================================================================

def test():
    print(getattr(BasicTransforms, "vflip"))

if __name__ == "__main__":
    test()



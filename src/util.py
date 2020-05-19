import sys
import torch
import pickle
import torchvision.transforms as tfs
from pathlib import Path

# helpers for train_model.py and train_gmaxup.py ========================================================
# =======================================================================================================

class Objects:
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    transform_pil_image_to_tensor = tfs.Compose([tfs.ToTensor(), tfs.Normalize((0.5,), (0.5,))])

class Constants:
    cifar10_x = 32
    cifar10_y = 32
    cifar10_channels = 3

    batch_size = 64
    learning_rate = 1e-3
    out_channels = 10

    randaugment_n = 1
    randaugment_m = 2

    dataset_str = "cifar10"
    

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
    pass

if __name__ == "__main__":
    test()



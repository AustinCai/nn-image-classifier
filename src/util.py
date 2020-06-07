import torch
import pickle
import torchvision.transforms as transforms

import augmentations

# helpers for train_model.py and train_gmaxup.py ========================================================
# =======================================================================================================

class Objects:
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Constants:
    save_str = "" # will be set before a log is saved 

    # training hyperparameters 
    batch_size = 128
    learning_rate = 1e-3
    model_str = "best_cnn"

    randaugment_n = 3
    randaugment_m = 4

    # dataset info 
    dataset_str = "cifar10"
    cifar10_dim = (32, 32, 3)
    out_channels = 10


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


# def pickle_load(path):
#     with open(path, 'rb') as handle:
#         return pickle.load(handle)

# helpers for train_gmaxup.py ===========================================================================
# =======================================================================================================

class GMaxupConsts:
    dataset_batches_dict = {
        "cifar-10-batches-py": ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    }
    samples_per_batch = 10000
    trans_magnitude = 30
    save_str = "untitled_save_str"


# def pickle_save(data, folder_path, identifier, data_str):
#     with open(folder_path / "{}-{}".format(data_str, identifier), 'wb') as data_file:
#         pickle.dump(data, data_file, protocol=pickle.HIGHEST_PROTOCOL)


# helpers for train_gmaxup.py ===========================================================================
# =======================================================================================================


# helpers for testing ===================================================================================
# =======================================================================================================

def test():
    print(getattr(BasicTransforms, "vflip"))

if __name__ == "__main__":
    test()



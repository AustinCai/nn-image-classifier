import torch
import torchvision
import torchvision.transforms as tfs
from pathlib import Path
import pickle
from torch.utils.data import Dataset

import util
from util import Constants
from util import Objects
from util import BasicTransforms
import augmentations

# helpers for train_model.py and train_gmaxup.py ========================================================
# =======================================================================================================

# helpers for train_model.py ============================================================================
# =======================================================================================================

def build_cifar10_ds(dataset_root_path="saved_data", 
    train_transform=BasicTransforms.pil_image_to_tensor, 
    test_transform=BasicTransforms.pil_image_to_tensor):
    '''Loads and returns training and test datasets, applying the provided 
    transform functions. Because cifar10 is a built-in dataset, only the 
    dataset root path must be specified. 
    '''
    train_ds = torchvision.datasets.CIFAR10(
        Path(__file__).parent.parent.parent / dataset_root_path, 
        train=True, transform=train_transform, download=False)
    valid_test_ds = torchvision.datasets.CIFAR10(
        Path(__file__).parent.parent.parent / dataset_root_path, 
        train=False, transform=test_transform, download=False)
    return train_ds, valid_test_ds


def build_mnist_ds(dataset_root_path="saved_data", 
    train_transform=BasicTransforms.pil_image_to_tensor, 
    test_transform=BasicTransforms.pil_image_to_tensor):
    '''Loads and returns training and test datasets, applying the provided 
    transform functions. Because mnist is a built-in dataset, only the
    dataset root path must be specified. 
    '''
    train_ds = torchvision.datasets.MNIST(
        Path(__file__).parent.parent.parent / dataset_root_path, 
        train=True, transform=train_transform, download=False)
    valid_test_ds = torchvision.datasets.MNIST(
        Path(__file__).parent.parent.parent / dataset_root_path, 
        train=False, transform=test_transform, download=False)
    return train_ds, valid_test_ds


def build_custom_cifar10_ds(dataset_path, 
    test_transform=BasicTransforms.pil_image_to_tensor):

    with open(Path(__file__).parent.parent.parent / dataset_path, 'rb') as handle:
        train_ds = pickle.load(handle)

    _, valid_test_ds = build_cifar10_ds(test_transform = test_transform)

    return train_ds, valid_test_ds

def build_dl(augmentation_str, dataset_str, shuffle=True, verbose=False): 
    '''Constructs and loads training and test dataloaders, which can be iterated 
    over to return one batch at a time. 
    '''
    transform = BasicTransforms.pil_image_to_tensor if augmentation_str == "none" \
        else getattr(BasicTransforms, augmentation_str)

    # built-in datasets
    if dataset_str == "mnist":
        train_ds, valid_test_ds = build_mnist_ds("saved_data", transform)           
    elif dataset_str == "cifar10":
        train_ds, valid_test_ds = build_cifar10_ds("saved_data", transform)

    # custom datasets
    elif "gmaxup_cifar" in dataset_str:
        train_ds, valid_test_ds = build_custom_cifar10_ds(dataset_str)
    else:
        raise Exception("Invalid dataset path {}".format(dataset_str))

    valid_ds, test_ds = torch.utils.data.random_split(
        valid_test_ds, [int(len(valid_test_ds)/2), int(len(valid_test_ds)/2)])
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=Constants.batch_size, shuffle=shuffle, drop_last=True)
    valid_dl = torch.utils.data.DataLoader(
        valid_ds, batch_size=Constants.batch_size, shuffle=shuffle, drop_last=True)
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=Constants.batch_size, shuffle=shuffle, drop_last=True)
    
    if verbose:
        print("In data_loading.build_dl() with dataset \'{}\' and augmentation \'{}\'.".format(
            dataset_str, augmentation_str))
        print("    len(train_ds): {}, len(valid_ds): {}, len(test_ds): {}.".format(
            len(train_ds), len(valid_ds), len(test_ds)))
        print("    len(train_dl): {}, len(valid_dl): {}, len(test_dl): {}.".format(
            len(train_dl), len(valid_dl), len(test_dl)))
        
    return train_dl, valid_dl, test_dl


class WrappedDataLoader:
    '''Wrapper that applies func to the wrapped dataloader.''' 
    def __init__(self, dataloader, reshape=False):
        self.dataloader = dataloader
        self.reshape = reshape
        self.dev = Objects.dev

    def __len__(self):
        return len(self.dataloader)

    def __next__(self):
        return self.next()

    def __iter__(self):
        for batch in iter(self.dataloader):
            x_batch, y_batch = batch
            if self.reshape:
                yield (x_batch.view(x_batch.shape[0], -1).to(self.dev), y_batch.to(self.dev))
            else:
                yield(x_batch.to(self.dev), y_batch.to(self.dev))


def wrap_dl(train_dl, valid_dl, test_dl, reshape, verbose=False):
    '''Creates two versions of training and test dataloaders: one the resizes 
    inputs and one that doesn't. The resized inputs are passed to the model,
    while the un-resized inputs are displayed as images on tensorboard. 
    '''
    train_dl = WrappedDataLoader(train_dl, reshape=reshape)
    valid_dl = WrappedDataLoader(valid_dl, reshape=reshape)
    test_dl = WrappedDataLoader(test_dl, reshape=reshape)

    if verbose:
        print("In data_loading.wrap_dl().")
        print("    reshape: {}".format(reshape))
        print("    respective lengths of train_dl, valid_dl, test_dl: " 
              + "{}, {}, {}.".format(len(train_dl), len(valid_dl), len(test_dl)))

    return train_dl, valid_dl, test_dl


def build_wrapped_dl(augmentation, dataset, verbose=False):
    train_dl, valid_dl, test_dl = build_dl(
        augmentation, dataset, verbose=verbose)
    train_dlr, valid_dlr, test_dlr = wrap_dl(
        train_dl, valid_dl, test_dl, not "cnn" in Constants.model_str, verbose)

    return train_dlr, valid_dlr, test_dlr


# helpers for train_gmaxup.py ===========================================================================
# =======================================================================================================

def load_data_by_path(path):
    f = open(path, 'rb')
    data_label_dict = pickle.load(f, encoding='bytes')
    f.close()

    images = data_label_dict[b'data'] # img = 10000 x 3072
    labels = data_label_dict[b'labels']

    return images, labels


class DatasetFromTupleList(Dataset):
    def __init__(self, tuple_list):
        self.samples = tuple_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# helpers for testing ===================================================================================
# =======================================================================================================

def test():
    pass


if __name__ == "__main__":
    test()
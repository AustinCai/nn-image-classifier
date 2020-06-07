import matplotlib
matplotlib.use('tkagg')  # Or any other X11 back-end

import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
import datetime

from util import Constants


# helpers for train_model.py and train_gmaxup.py ========================================================
# =======================================================================================================


def show_images(writer, images, batch_size, title="Images", verbose=False):
    '''Displays the input images passed in through train_dl, both to the console
    and to tensorboard. 
    '''
    start_time = time.time()

    images = images.view(batch_size, Constants.cifar10_dim[2], \
        Constants.cifar10_dim[0], Constants.cifar10_dim[1]) 
    norm_min, norm_max = -1, 1
    img_grid = torchvision.utils.make_grid(images, normalize=True, range=(norm_min, norm_max))
    if verbose: 
        print("In display.show_images(title={}).".format(title))
        print("    images.shape: {}.".format(images.shape))
        print("    img_grid.shape: {}.".format(img_grid.shape))

    # format_and_show(img_grid, one_channel=False)
    writer.add_image(title, img_grid)

    print("    display.show_images() completed in {} seconds.".format(time.time() - start_time))


# helpers for train_model.py ============================================================================
# =======================================================================================================

def build_model_save_str(args):
    optional_tokens = [] 
    if args.load_model:
        optional_tokens.append("pretrained_{}e".format(pretrained_epochs))
    if args.name:
        optional_tokens.append(args.name)
    optional_str = ""
    if len(optional_tokens):
        for token in optional_tokens:
            optional_str += "{}-".format(token)

    return '{}-{}e-{}-{}{}'.format(
        Constants.model_str, 
        args.epochs, 
        "gmaxup" if "gmaxup_cifar" in args.dataset else args.augmentation, # augmentation string
        optional_str, # optional string 
        datetime.datetime.now().strftime("%-m.%d.%y-%H:%M"))


def format_and_show(img, one_channel=False):
    '''Resizes the passed img parameter so that it can be accepted by 
    plt.imshow(). 
    '''
    if one_channel:
        img = img.mean(dim=0)
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        channel_idx, row_idx, col_idx = 0, 1, 2
        plt.imshow(np.transpose(npimg, (row_idx, col_idx, channel_idx)))


def show_graph(writer, model, images):
    start_time = time.time()

    # hack, only one of these works, will investigate why later
    try:
        writer.add_graph(model, images) # works for CNN
    except:
        writer.add_graph(model, images.view(Constants.batch_size, -1)) # works for NN 
    print("    display.show_graph() completed in {} seconds.".format(time.time() - start_time))   


def write_epoch_stats(writer, epoch, validation_acc, validation_loss, train_acc, train_loss):
    writer.add_scalar('Accuracy/Validation', validation_acc, epoch)
    writer.add_scalar('Loss/Validation', validation_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalars("Validation vs. Train Accuracy", 
        {"validation": validation_acc, "train": train_acc}, epoch)


def print_final_model_stats(train_acc, validation_acc, test_acc):
    print("Final model statistics:")
    print("    training accuracy: {}".format(train_acc))
    print("    validation accuracy: {}".format(validation_acc))
    print("    train/val difference: {}".format(train_acc-validation_acc))
    print("    test accuracy: {}".format(test_acc))

# helpers for train_gmaxup.py ===========================================================================
# =======================================================================================================

def log_image_and_transformations(writer, x_aug_tensor, c, used_augment_mag_tuples, loss, sample_num):
    '''For each candidate augmentation sequence, write the transformed image to tensorboard and print
    it corresponding loss info.'''

    show_images(writer, x_aug_tensor, 1, 
        title="Image_{}-{}_{}-{}_{}-Loss_{}".format(
            c, 
            used_augment_mag_tuples[0][0], 
            used_augment_mag_tuples[0][1], 
            used_augment_mag_tuples[1][0], 
            used_augment_mag_tuples[1][1], 
            round(loss,2)))
    print("operation_1: {}_{}".format(used_augment_mag_tuples[0][0], used_augment_mag_tuples[0][1]))
    print("operation_2: {}_{}".format(used_augment_mag_tuples[1][0], used_augment_mag_tuples[1][1]))
    print("loss: {}".format(round(loss,2)))
    print("img_id: {}".format(sample_num+c))
    print("===================================")


def build_gmaxup_save_str(args):
    '''Build a unique identifying string for this run and settings.'''

    optional_tokens = []
    if args.name:
        optional_tokens.append(args.name)
    optional_str = ""
    if len(optional_tokens):
        for token in optional_tokens:
            optional_str += "{}-".format(token)

    return "gmaxup_{}-{}-{}l-{}s-{}c-{}{}".format(
        args.dataset, args.range, args.layers, args.setsize, args.choices, optional_str, # optional string 
        datetime.datetime.now().strftime("%-m.%d.%y-%H:%M"))


def print_augment_stats(augment_stats):
    for augment_str, stats in augment_stats.items():
        print("{}:".format(augment_str))
        print("    incorrect/count: {}/{}".format(stats["incorrect_preds"], stats["count"]))
        print("    average magnitude: {}".format(stats["average_mag"]))
        print("    average loss: {}".format(stats["average_loss"]))


# helpers for testing ===================================================================================
# =======================================================================================================

def test():
    pass


if __name__ == "__main__":
    test()
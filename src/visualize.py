import matplotlib
matplotlib.use('tkagg')  # Or any other X11 back-end

import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time

from util import Constants


# helpers for train_model.py and train_gmaxup.py ========================================================
# =======================================================================================================

# helpers for train_model.py ============================================================================
# =======================================================================================================

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
        print("In visualize.show_images(title={}).".format(title))
        print("    images.shape: {}.".format(images.shape))
        print("    img_grid.shape: {}.".format(img_grid.shape))

    # format_and_show(img_grid, one_channel=False)
    writer.add_image(title, img_grid)

    print("    visualize.show_images() completed in {} seconds.".format(time.time() - start_time))


def show_graph(writer, model, images):
    start_time = time.time()

    # hack, only one of these works, will investigate why later
    try:
        writer.add_graph(model, images) # works for CNN
    except:
        writer.add_graph(model, images.view(Constants.batch_size, -1)) # works for NN 
    print("    visualize.show_graph() completed in {} seconds.".format(time.time() - start_time))   



def write_epoch_stats(writer, epoch, validation_acc, validation_loss, train_acc, train_loss):
    writer.add_scalar('Accuracy/Validation', validation_acc, epoch)
    writer.add_scalar('Loss/Validation', validation_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)
    writer.add_scalar('Loss/Train', train_loss, epoch)

def write_epoch_statistics(writer, epoch, epoch_stats):
    writer.add_scalar('Accuracy/Train', 
        epoch_stats["accuracy"],
        global_step=epoch+1)
    writer.add_scalar('Loss/Train', 
        epoch_stats["loss"], 
        global_step=epoch+1)   
    
    if (epoch): # validation of first epoch is undefined
        writer.add_scalar('Loss/Validation', 
            epoch_stats["validation_loss"], 
            global_step=epoch+1) 

    if isinstance(epoch_stats["validation_accuracy"], float): # skip if "NA"
        writer.add_scalars("Validation vs. Train Accuracy", 
            {"validation": epoch_stats["validation_accuracy"], 
            "train": epoch_stats["accuracy"]}, 
            global_step=epoch+1)
        writer.add_scalar('Accuracy/Validation', 
            epoch_stats["validation_accuracy"], 
            global_step=epoch+1)

def normalize_cmatrix(cmatrix_test, norm_max):
    return np.floor(((norm_max-.0001)/cmatrix_test.max())*cmatrix_test).astype(int) + 1


def print_save_cmatrix(writer, y_batch_test, yh_batch_test):
    '''Generates a confusion matrix, which is logged to console and saved to
    tensorboard as an image. 
    '''
    cmatrix_test = confusion_matrix(y_batch_test, yh_batch_test)
    cmatrix_test_toprint = normalize_cmatrix(cmatrix_test, 255)
    writer.add_image("Confusion Matrix", cmatrix_test_toprint, dataformats='HW')
    writer.close()
    print(cmatrix_test)


def print_final_model_stats(last_epoch_stats, accuracy):
    print("Final model statistics:")
    print("    training accuracy: {}".format(last_epoch_stats["accuracy"]))
    print("    validation accuracy: {}".format(last_epoch_stats["validation_accuracy"]))
    if isinstance(last_epoch_stats["validation_accuracy"], float):
        print("    train/val difference: {}".format(
            last_epoch_stats["accuracy"] - last_epoch_stats["validation_accuracy"]))
    print("    test accuracy: {}".format(accuracy))

# helpers for train_gmaxup.py ===========================================================================
# =======================================================================================================


# helpers for testing ===================================================================================
# =======================================================================================================

def test():
    pass


if __name__ == "__main__":
    test()
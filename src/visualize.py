#Visualize
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time

from util import Constants

# def seed(s):
#     np.random.seed(s)

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


def show_images(writer, images, title="Images", verbose=False):
    '''Displays the input images passed in through train_dl, both to the console
    and to tensorboard. 
    '''
    start_time = time.time()

    images = images.view(Constants.batch_size, Constants.cifar10_channels, \
        Constants.cifar10_x, Constants.cifar10_y) 
    norm_min, norm_max = -1, 1
    img_grid = torchvision.utils.make_grid(images, normalize=True, range=(norm_min, norm_max))
    if verbose: 
        print("In visualize.show_images(title={}).".format(title))
        print("    images.shape: {}.".format(images.shape))
        print("    img_grid.shape: {}.".format(img_grid.shape))

    format_and_show(img_grid, one_channel=False)
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
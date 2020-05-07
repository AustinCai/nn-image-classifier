#Visualize
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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
        CHANNEL_IDX, ROW_IDX, COL_IDX = 0, 1, 2
        plt.imshow(np.transpose(npimg, (ROW_IDX, COL_IDX, CHANNEL_IDX)))


def show_inputs(writer, raw_dl, model, train_batch_size, verbose):
    '''Displays the input images passed in through train_dl, both to the console
    and to tensorboard. 
    '''
    NORM_MIN, NORM_MAX = -1, 1
    images, labels = iter(raw_dl).__next__()
    print("images.size(): {}".format(images.size()))
    img_grid = torchvision.utils.make_grid(
        images, normalize=True, range=(NORM_MIN, NORM_MAX))
    if verbose: 
        print("In show_inputs().")
        print("    images.shape: {}.".format(images.shape))
        print("    img_grid.shape: {}.".format(img_grid.shape))

    format_and_show(img_grid, one_channel=False)
    # hack, only one of these works, will investigate why later
    writer.add_image('four_digit_mnist_images', img_grid)
    writer.add_graph(model, images) # works for CNN
    # writer.add_graph(model, images.view(train_batch_size, -1))
    writer.close()


def normalize_cmatrix(cmatrix_test, norm_max):
    return np.floor(
        ((norm_max-.0001)/cmatrix_test.max())*cmatrix_test).astype(int) + 1


def print_save_cmatrix(writer, y_batch_test, yh_batch_test):
    '''Generates a confusion matrix, which is logged to console and saved to
    tensorboard as an image. 
    '''
    cmatrix_test = confusion_matrix(y_batch_test, yh_batch_test)
    cmatrix_test_toprint = normalize_cmatrix(cmatrix_test, 255)
    writer.add_image("Confusion Matrix", cmatrix_test_toprint, dataformats='HW')
    writer.close()
    print(cmatrix_test)
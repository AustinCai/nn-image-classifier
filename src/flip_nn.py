import math
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn 
import data_loading
from util import Constants
from util import Objects
from pathlib import Path
import display


def build_vflip_matrix(img_dim, flat_dim):
	'''Creates a matrix to parameterize a vertical flip.'''

	vflip_matrix_list = []
	for r in range(flat_dim):
		row = [0. for _ in range(flat_dim)]
		row[flat_dim - img_dim*(math.floor(r/img_dim)+1) + r%img_dim] = 1.
		vflip_matrix_list.append(row)
	return torch.FloatTensor(vflip_matrix_list).to(Objects.dev)


def build_hflip_matrix(img_dim, flat_dim):
	'''Creates a matrix to parameterize a horizontal flip.'''

	hflip_matrix_list = []
	for r in range(flat_dim):
	    row = [0. for _ in range(flat_dim)]
	    row[img_dim*(math.floor(r/img_dim)+1) - r%img_dim - 1] = 1.
	    hflip_matrix_list.append(row)
	return torch.FloatTensor(hflip_matrix_list).to(Objects.dev)


def load_singleton_batch_from_cifar10():
	'''Returns a single cifar10 image.'''

	baseline_transforms = data_loading.init_baseline_transforms()
	train_dl, valid_dl, test_dl = data_loading.build_dl(
		baseline_transforms, "none", "cifar10")
	train_dlr, valid_dlr, test_dlr = data_loading.wrap_dl(
	    train_dl, valid_dl, test_dl, True)

	x_batch, y_batch = iter(train_dlr).__next__()
	image = x_batch[0]

	return image.view(1, -1)


def load_singleton_batch_dummy():
	'''Returns a dummy 3*3 "image" for testing.'''

	image_list = 3*[i for i in range(1,10)]
	return torch.FloatTensor(image_list).view(1,-1).to(Objects.dev)


def main():
	img_dim = Constants.cifar10_dim[0]
	flat_dim = Constants.cifar10_dim[0] * Constants.cifar10_dim[1] * Constants.cifar10_dim[2]

	print("Using GPU: {}".format(torch.cuda.is_available()))
	writer = SummaryWriter(Path(__file___).parent.parent / "runs" / "flip_nn_test")

	model = torch.nn.Sequential(nn.Linear(flat_dim, flat_dim, bias=False))

	matrix = build_vflip_matrix(img_dim, flat_dim)
	with torch.no_grad():
	    model[0].weight = nn.Parameter(matrix)

	singleton_batch = load_singleton_batch_from_cifar10()
	display.show_images(writer, singleton_batch, 1, "original")

	singleton_batch_flipped = model(singleton_batch)
	display.show_images(writer, singleton_batch_flipped, 1, "flipped")


if __name__ == "__main__":
    main()




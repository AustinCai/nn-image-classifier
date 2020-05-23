import math
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn 
import data_loading
import util 
from util import Constants
from pathlib import Path
import visualize

def main():
	# https://discuss.pytorch.org/t/how-to-manually-set-the-weights-in-a-two-layer-linear-model/45902
	flattened_dim = Constants.cifar10_x * Constants.cifar10_y * Constants.cifar10_channels
	original_side_len = Constants.cifar10_x

	print("Using GPU: {}".format(torch.cuda.is_available()))
	writer = SummaryWriter(Path("runs") / "flip_nn_test")

	model = torch.nn.Sequential(nn.Linear(flattened_dim, flattened_dim, bias=False))

	hflip_matrix_list = []
	for r in range(flattened_dim):
	    row = [0. for _ in range(flattened_dim)]
	    row[original_side_len + original_side_len*math.floor(r/original_side_len) - r%original_side_len - 1] = 1.
	    hflip_matrix_list.append(row)

	hflip_matrix_tensor = torch.FloatTensor(hflip_matrix_list)

	with torch.no_grad():
	    model[0].weight = nn.Parameter(hflip_matrix_tensor.to(util.Objects.dev))

	baseline_transforms = data_loading.init_baseline_transforms()
	train_dl, valid_dl, test_dl = data_loading.build_dl(baseline_transforms, "none", "cifar10")
	train_dlr, valid_dlr, test_dlr = data_loading.wrap_dl(
	    train_dl, valid_dl, test_dl, True)

	x_batch, y_batch = iter(train_dlr).__next__()
	image = x_batch[0]
	print(image.shape)

	singleton_batch = image.view(1, -1)
	visualize.show_images(writer, singleton_batch, 1, "original")

	print(singleton_batch.shape)
	singleton_batch_flipped = model(singleton_batch)
	print(singleton_batch_flipped.shape)

	# visualize 
	visualize.show_images(writer, singleton_batch_flipped, 1, "flipped")

if __name__ == "__main__":
    main()




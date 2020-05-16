import sys

class Constants:
    cifar10_x = 32
    cifar10_y = 32
    cifar10_channels = 3
    batch_size = 64
    out_channels = 3072

    # constant for dev

#Util
def assert_params(run_specifications, dataset):
    assert (dataset in {"mnist", "cifar10"}), "Invalid dataset specification of {}".format(dataset)

    valid_specs = {
        "model_str": {"small_nn", "large_nn", "linear", "small_cnn", "best_cnn"},
        "augmentation": {"none", "vflip", "hflip", "contrast", "random"},
        "optimizer": {"sgd", "adam"}
    }

    for i, run_spec in enumerate(run_specifications):
        for spec_to_check, valid_values in valid_specs.items():
            assert (run_spec[spec_to_check] in valid_values), \
                "Invalid {} specification of {}".format(spec_to_check, run_spec[spec_to_check])


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
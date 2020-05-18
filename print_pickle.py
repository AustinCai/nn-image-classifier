import sys
import argparse
import pickle

def get_args(arguments):
    '''Parse the arguments passed via the command line.
    '''
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-f', '--file', help = 'Print saved info in given pickle file', type=str)
    args = parser.parse_args(arguments)
    return args


def main(args):

    with open(args.file, 'rb') as handle:
        b = pickle.load(handle)

    print(b)
    return b


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    main(args)
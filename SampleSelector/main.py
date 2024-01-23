import sys

from argparse import ArgumentParser

from SampleSelector.methods import iter_files

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)

    args = parser.parse_args()

    iter_files(args.data_dir, sys.argv)

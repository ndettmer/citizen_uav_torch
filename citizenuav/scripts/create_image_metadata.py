from argparse import ArgumentParser
from citizenuav.processes import create_image_metadata


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--classes", type=str, nargs='+', required=False)
    parser.add_argument("--debug", type=bool, required=False, default=False)

    args = parser.parse_args()
    dict_args = vars(args)
    create_image_metadata(**dict_args)

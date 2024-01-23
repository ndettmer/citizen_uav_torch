from argparse import ArgumentParser
from citizenuav.processes import offline_augmentation_regression_data

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--target_n", type=int, required=True)
    parser.add_argument("--debug", type=bool, required=False, default=False)

    args = parser.parse_args()
    dict_args = vars(args)
    offline_augmentation_regression_data(**dict_args)

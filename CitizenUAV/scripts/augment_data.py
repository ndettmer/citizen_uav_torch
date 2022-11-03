from argparse import ArgumentParser
from CitizenUAV.processes import offline_augmentation

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--target_n", type=int, required=True)
    parser.add_argument("--subdirs", type=str, nargs='+', required=True)
    parser.add_argument("--distances", type=str, required=False, default=False)

    args = parser.parse_args()
    dict_args = vars(args)
    offline_augmentation(**dict_args)

from argparse import ArgumentParser
from CitizenUAV.data import offline_augmentation

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--target_n", type=int, required=True)

    args = parser.parse_args()
    dict_args = vars(args)
    offline_augmentation(**dict_args)

from argparse import ArgumentParser
from CitizenUAV.processes import extend_metadata


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--consider_augmented", type=bool)

    args = parser.parse_args()
    dict_args = vars(args)
    extend_metadata(**dict_args)

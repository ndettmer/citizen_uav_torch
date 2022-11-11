from argparse import ArgumentParser
from CitizenUAV.processes import create_gim_metadata


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--classes", type=str, nargs='+', required=False)
    parser.add_argument("--debug", type=bool, required=False, default=False)

    args = parser.parse_args()
    dict_args = vars(args)
    create_gim_metadata(**dict_args)

from argparse import ArgumentParser, ArgumentTypeError
from CitizenUAV.data import download_data


MIN_YEAR = 2010
MAX_YEAR = 2024


def obs_year_type(x):
    if x is None:
        return MIN_YEAR
    x = int(x)
    if x < MIN_YEAR:
        raise ArgumentTypeError(f"Observations before {MIN_YEAR} are not supported.")
    if x >= MAX_YEAR:
        raise ArgumentTypeError(f"Observations after {MAX_YEAR} are not supported.")
    return x


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--species", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_images", type=int)
    parser.add_argument("--min_year", type=obs_year_type, help=f"Integer between {MIN_YEAR} and {MAX_YEAR}")

    args = parser.parse_args()
    dict_args = vars(args)
    download_data(**dict_args)

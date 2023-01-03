from CitizenUAV.processes import predict_geotiff
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--window_size", type=int, required=False, default=128)
    parser.add_argument("--stride", type=int, required=False, default=1)
    parser.add_argument("--gpu", type=bool, required=False, default=True)
    parser.add_argument("--batch_size", type=int, required=False, default=1)

    args = parser.parse_args()
    dict_args = vars(args)
    predict_geotiff(**dict_args)

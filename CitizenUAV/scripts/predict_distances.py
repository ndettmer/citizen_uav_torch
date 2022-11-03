from CitizenUAV.processes import predict_distances
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--img_size", type=int, choices=[2 ** x for x in range(6, 10)], required=False, default=256)
    parser.add_argument("--train_min", type=float, required=True)
    parser.add_argument("--train_max", type=float, required=True)
    parser.add_argument("--species", type=str, nargs='+', required=False)
    parser.add_argument("--batch_size", type=int, required=False, default=1)
    parser.add_argument("--gpu", type=bool, required=False, default=True)
    parser.add_argument("--overwrite", type=bool, required=False, default=False)

    args = parser.parse_args()
    dict_args = vars(args)
    predict_distances(**dict_args)

from argparse import ArgumentParser

from plyer import notification

from citizenuav.processes import predict_inat

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, required=False, default=128)
    parser.add_argument("--min_distance", type=float, required=False, default=None)
    parser.add_argument("--gpu", type=bool, required=False, default=None)
    parser.add_argument("--batch_size", type=int, required=False, default=1)
    parser.add_argument("--normalize", type=bool, required=False, default=False)
    parser.add_argument(
        "--model_class",
        type=str,
        required=False,
        default="InatSequentialClassifier",
        choices=["InatSequentialClassifier", "InatMogaNetClassifier"],
    )

    args = parser.parse_args()
    dict_args = vars(args)
    predict_inat(**dict_args)

    notification.notify(title="Inat Predictions", message="Predictions done.")

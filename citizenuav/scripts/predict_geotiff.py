from citizenuav.processes import predict_geotiff
from citizenuav.io import write_params

from argparse import ArgumentParser
import os

from plyer import notification


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=False, default=None)
    parser.add_argument("--window_size", type=int, required=False, default=128)
    parser.add_argument("--stride", type=int, required=False, default=1)
    parser.add_argument("--gpu", type=bool, required=False, default=True)
    parser.add_argument("--batch_size", type=int, required=False, default=os.cpu_count())
    parser.add_argument("--normalize", type=bool, required=False, default=False)
    parser.add_argument("--means", nargs=3, type=float, required=False, default=None)
    parser.add_argument("--stds", nargs=3, type=float, required=False, default=None)
    parser.add_argument("--probabilities", type=bool, required=False, default=False)
    parser.add_argument("--pred_size", type=int, required=False, default=None)
    parser.add_argument("--model_class", type=str, required=False, default='InatSequentialClassifier',
                        choices=['InatSequentialClassifier', 'InatMogaNetClassifier'])
    parser.add_argument("--debug", type=bool, required=False, default=False)
    args = parser.parse_args()
    dict_args = vars(args)
    write_params(args.result_dir, dict_args, "predict_geotiff")
    predict_geotiff(**dict_args)

    notification.notify(
        title="GeoTiff Prediction",
        message="Done predicting."
    )

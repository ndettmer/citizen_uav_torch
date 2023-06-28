from CitizenUAV.processes import visualize_class_features
from CitizenUAV.io_utils import write_params
from argparse import ArgumentParser
from plyer import notification


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--preds_path", type=str)
    parser.add_argument("--out_dir", type=str, required=False)
    parser.add_argument("--samples_per_class", type=int, required=False, default=5)
    parser.add_argument("--model_class", type=str, required=False,
                        choices=['InatSequentialClassifier', 'InatMogaNetClassifier'],
                        default='InatSequentialClassifier')
    parser.add_argument("--nt_samples", type=int, required=False, default=10,
                        help="Number of noise tunnel samples to be used in integrated gradients.")

    args = parser.parse_args()
    dict_args = vars(args)
    write_params(args.out_dir, dict_args, "visualize_feature_attribution")
    visualize_class_features(**dict_args)

    notification.notify(
        title="Visualize Feature Attribution",
        message="Done visualizing."
    )

from CitizenUAV.processes import visualize_confusion_resnet
from CitizenUAV.io_utils import write_params
from argparse import ArgumentParser
from plyer import notification
from typing import Optional


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--preds_path", type=str)
    parser.add_argument("--out_dir", type=str, required=False)
    parser.add_argument("--n_samples", type=int, required=False, default=None)

    args = parser.parse_args()
    dict_args = vars(args)
    write_params(args.out_dir, dict_args, "visualize_feature_attribution")
    visualize_confusion_resnet(**dict_args)

    notification.notify(
        title="Visualize Confusion ResNet",
        message="Done visualizing."
    )

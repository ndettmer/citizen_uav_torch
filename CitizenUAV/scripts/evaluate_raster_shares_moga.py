from argparse import ArgumentParser
from datetime import datetime
from plyer import notification
import os
import numpy as np
import pandas as pd

from CitizenUAV.models import InatMogaNetClassifier
from CitizenUAV.data import MixedDataModule
from CitizenUAV.io_utils import write_params
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt

from CitizenUAV.processes import predict_geotiff, pixel_conf_mat


def get_model_checkpoint_in_dir(dir_path):
    pass

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str, default='./lightning_logs')
    parser.add_argument("--patience", type=int, default=-1)
    parser.add_argument("--min_delta", type=float, default=0)
    parser.add_argument("--n_runs", type=int, help="Number of runs per value for n_supplement.")
    parser.add_argument("--n_supplement_min", type=int, help="Maximum number of raster supplement samples per class.", required=False, default=0)
    parser.add_argument("--n_supplement_max", type=int, help="Maximum number of raster supplement samples per class.")
    parser.add_argument("--n_supplement_step", type=int,
                        help="Stipe size for increasing number of raster supplement samples per class.")
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--pred_dataset_paths", type=str, nargs='+', required=True)
    parser.add_argument("--pred_shape_dirs", type=str, nargs='+', required=True)
    parser.add_argument("--pred_size", type=int, default=16)
    parser.add_argument("--pred_batch_size", type=int, required=False, default=os.cpu_count())
    parser.add_argument("--probabilities", type=bool, required=False, default=True)
    parser.add_argument("--debug", type=bool, required=False, default=False)
    parser.add_argument("--seed", type=int, default=np.random.rand())

    parser = MixedDataModule.add_dm_specific_args(parser)
    parser = InatMogaNetClassifier.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    write_params(args.log_dir, vars(args), 'train_classifier')

    data_dir = args.data_dir
    log_dir = args.log_dir
    seed_everything(args.seed, workers=True)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # if no subset is defined take all present species
    args.n_classes = len(next(os.walk(args.inat_dir))[1])

    dict_args = vars(args)
    dict_args['gpu'] = dict_args['accelerator'] == 'cuda'
    dict_args['model_class'] = 'InatMogaNetClassifier'
    dict_args['window_size'] = dict_args['img_size']
    dict_args['result_dir'] = dict_args['log_dir']

    f1_df = pd.DataFrame(columns=['version', 'n_supplement', 'May F1', 'June F1'], dtype=float)
    f1_df.version = f1_df.version.astype(int)
    f1_df.n_supplement = f1_df.n_supplement.astype(int)

    for n_supplement in range(args.n_supplement_min, args.n_supplement_max, args.n_supplement_step):
        dict_args['n_raster_samples_per_class'] = n_supplement

        for i_run in range(args.n_runs):

            tb_logger = TensorBoardLogger(save_dir=log_dir)
            dm = MixedDataModule(**dict_args)
            model = InatMogaNetClassifier(**dict_args)
            train_classes = dm.ds.classes

            callbacks = []
            if args.patience > 0:
                callbacks.append(
                    EarlyStopping(
                        monitor="val_cce",
                        mode="min",
                        patience=args.patience,
                        verbose=True,
                        min_delta=args.min_delta
                    )
                )

            trainer = Trainer.from_argparse_args(args, logger=tb_logger)

            trainer.fit(model, dm)

            trainer.test(ckpt_path='best', dataloaders=dm.test_dataloader())

            version_no = len(os.listdir(os.path.join(dict_args['log_dir'], 'lightning_logs'))) - 1
            model_dir = os.path.join(dict_args['log_dir'], "lightning_logs", f"version_{version_no}", "checkpoints")
            dict_args['model_path'] = os.path.join(model_dir, os.listdir(model_dir)[0])

            f1_row = [version_no, n_supplement]

            for pred_ds_path, pred_shape_dir in zip(dict_args['pred_dataset_paths'], dict_args['pred_shape_dirs']):
                dict_args['dataset_path'] = pred_ds_path
                prob_mask, predfile = predict_geotiff(**dict_args)

                plot_dir = os.path.join(dict_args['result_dir'], "plots")
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                prob_mask /= prob_mask.max()

                figure, axs = plt.subplots(1, 3, figsize=(15, 15))
                for i in range(3):
                    label = train_classes[i]
                    axs[i].imshow(prob_mask[i])
                    axs[i].set_title(label)
                plt.savefig(os.path.join(plot_dir, os.path.basename(predfile).replace('npy', 'png')))

                cm, df, f1 = pixel_conf_mat(dict_args['dataset_path'], pred_shape_dir, dict_args['data_dir'], predfile,
                                            None, result_dir=plot_dir)
                f1_row.append(f1)

            f1_df.loc[version_no] = f1_row

        f1_df.to_csv(
            os.path.join(
                dict_args['result_dir'],
                f"{datetime.now().strftime('%y-%m-%d_%H-%M')}_n{n_supplement}_{dict_args['backbone_model']}_f1_df.csv"
            ),
            index=False
        )

    notification.notify(
        title="Raster Share Evaluation",
        message="Done."
    )

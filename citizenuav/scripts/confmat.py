from citizenuav.processes import pixel_conf_mat

from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--shape_dir", type=str)
    parser.add_argument("--train_data_dir", type=str)
    parser.add_argument("--pred_file", type=str)
    parser.add_argument("--class_map", type=str,
                        help="Mapping from iNaturalist classes to raster data classes in JSON format")
    parser.add_argument("--crop_class", type=str, required=False, default=None)

    args = parser.parse_args()
    dict_args = vars(args)
    pixel_conf_mat(**dict_args)

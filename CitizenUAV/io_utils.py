from dataclasses import dataclass, asdict
import os
import sys
from datetime import datetime
from typing import Tuple, Union, Optional
from pathlib import Path
import logging
import yaml
import numpy as np
from abc import ABC

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm


def get_pid_from_path(path):
    """
    Takes a path to an INaturalist image file and removes the directories and file extension leaving only the photo ID.
    :param path: Path to the file
    :return int: INaturalist photo ID
    """
    filename = os.path.splitext(os.path.basename(path))[0]
    return str(filename)


def read_inat_metadata(data_dir):
    """
    Reading routine for metadata.csv file for an iNaturalist dataset.
    :param data_dir: The directory the data and its corresponding metadata lie in.
    :return pd.DataFrame: The metadata DataFrame.
    """
    metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    metadata.photo_id = metadata.photo_id.astype(str)
    metadata.set_index('photo_id', inplace=True)
    metadata = metadata[~metadata.index.duplicated(keep='first')]
    if 'Unnamed: 0' in metadata:
        metadata.drop(columns=['Unnamed: 0'], inplace=True)
    if 'distance' in metadata.columns:
        metadata.distance = metadata.distance.astype(float)
    if 'broken' in metadata.columns:
        metadata.broken = metadata.broken.astype(bool)
    if 'path' in metadata.columns:
        metadata.path = metadata.path.astype(str)
    return metadata


def read_split_inat_metadata(data_dir: Union[str, Path], species: Optional[list[str]] = None):
    _, dirs, _ = next(os.walk(data_dir))
    if not len(dirs):
        raise FileNotFoundError(f"No metadata.csv or other subdirectories found in {data_dir}.")
    logging.info("Found the following species:\n" + "\n".join(dirs))

    if species is not None and species:
        dirs = [d for d in dirs if d in species]

    dfs = []
    cols = []
    for d in dirs:
        subdir = os.path.join(data_dir, d)
        if not os.path.exists(os.path.join(subdir, 'metadata.csv')):
            df = read_split_inat_metadata(subdir)
            df.species = d
        else:
            df = read_inat_metadata(subdir)
            if len(df.columns) > len(cols):
                cols = df.columns
        dfs.append(df)

    for df in dfs:
        for col in cols:
            if col not in df.columns:
                df[col] = pd.Series(dtype=object)

    combined = pd.concat(dfs)
    if 'image_okay' in combined.columns:
        combined.image_okay = combined.image_okay.astype(bool)
        combined.image_okay.fillna(False)
    return combined


def store_split_inat_metadata(metadata: pd.DataFrame, data_dir: Union[str, Path]):
    for spec in metadata.species.unique():
        df = metadata[metadata.species == spec]
        assert df.index.name == 'photo_id'
        df.reset_index(inplace=True)
        try:
            df.to_csv(os.path.join(data_dir, spec, 'metadata.csv'), index=False)
            df.to_csv(os.path.join(data_dir, spec, 'metadata_backup.csv'), index=False)
        except KeyboardInterrupt:
            df.to_csv(os.path.join(data_dir, spec, 'metadata.csv'), index=False)
            df.to_csv(os.path.join(data_dir, spec, 'metadata_backup.csv'), index=False)
            sys.exit()


def write_params(dest_dir: Union[str, Path], params: dict, func_name: Optional[str] = None):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    filename = os.path.join(dest_dir,
                            f'{datetime.now().strftime("%y-%m-%d_%H-%M")}{("_" + func_name) if func_name is not None else ""}_parameters.yml')
    with open(filename, 'w') as outfile:
        return yaml.dump(params, outfile)

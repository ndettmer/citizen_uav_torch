import shutil
from dataclasses import dataclass, asdict
import os
import sys
from datetime import datetime
from typing import Union, Optional
from pathlib import Path
import logging
import yaml

import pandas as pd


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
    metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'), low_memory=False)
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

    combined.dropna(subset=['species'], inplace=True)
    combined.species = combined.species.astype(str)

    if 'image_okay' in combined.columns:
        combined.image_okay = combined.image_okay.astype(bool)
        combined.image_okay.fillna(False)
    return combined


def store_split_inat_metadata(metadata: pd.DataFrame, data_dir: Union[str, Path]):
    classnames = next(os.walk(data_dir))[1]

    for spec in metadata.species.unique():
        df = metadata[metadata.species == spec]
        assert df.index.name == 'photo_id'
        df.reset_index(inplace=True)

        # determine saving path
        save_path = os.path.join(data_dir, spec)
        if not os.path.exists(save_path):
            path_found = False

            for classname in classnames:
                if not path_found:
                    class_dir = os.path.join(data_dir, classname)
                    subclassnames = next(os.walk(class_dir))[1]
                    if spec in subclassnames:
                        save_path = class_dir
                        break
                    for subclassname in subclassnames:
                        subclass_dir = os.path.join(class_dir, subclassname)
                        subsubclassnames = next(os.walk(subclass_dir))[1]
                        if spec in subsubclassnames:
                            save_path = subclass_dir
                            path_found = True
                            break

        try:
            df.to_csv(os.path.join(save_path, 'metadata.csv'), index=False)
            df.to_csv(os.path.join(save_path, 'metadata_backup.csv'), index=False)
        except KeyboardInterrupt:
            df.to_csv(os.path.join(save_path, 'metadata.csv'), index=False)
            df.to_csv(os.path.join(save_path, 'metadata_backup.csv'), index=False)
            sys.exit()


def write_params(dest_dir: Union[str, Path], params: dict, func_name: Optional[str] = None):
    dest_dir = os.path.join(dest_dir, 'run_parameters')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    filename = os.path.join(dest_dir,
                            f'{datetime.now().strftime("%y-%m-%d_%H-%M")}{("_" + func_name) if func_name is not None else ""}_parameters.yml')
    with open(filename, 'w') as outfile:
        return yaml.dump(params, outfile)


def empty_dir(target_dir):
    for filename in os.listdir(target_dir):
        file_path = os.path.join(target_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

import os
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
    metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    metadata.photo_id = metadata.photo_id.astype(str)
    metadata.set_index('photo_id', inplace=True)
    metadata = metadata[~metadata.index.duplicated(keep='first')]
    if 'Unnamed: 0' in metadata:
        metadata.drop(columns=['Unnamed: 0'], inplace=True)
    return metadata

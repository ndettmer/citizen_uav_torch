import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
import pyinaturalist as pin
from io import BytesIO
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
from typing import Optional


def download_data(species, output_dir, max_images=None):
    quality = "research"

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    metadata_path = os.path.join(output_dir, 'metadata.csv')
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)
        metadata.reset_index(drop=True, inplace=True)
        metadata.set_index('photo_id', inplace=True)
    else:
        metadata = pd.DataFrame(columns=['species', 'obs_id', 'n_photos', 'label'])
        metadata.index.name = 'photo_id'

    response = pin.get_observations(
        taxon_name=species,
        quality_grade=quality,
        photos=True,
        page='all'
    )
    obss = pin.Observations.from_json_list(response)

    if not len(obss):
        print(f"No observations found for species {species}")
        return False

    images_stored = 0

    for obs in tqdm(obss):

        n_photos = len(obs.photos)

        for i, photo in enumerate(obs.photos):

            spec_dir = os.path.join(output_dir, species)
            if not os.path.isdir(spec_dir):
                os.makedirs(spec_dir)
            filename = f'{photo.id}.png'
            img_path = os.path.join(spec_dir, filename)
            if os.path.exists(img_path) and photo.id in metadata.index:
                continue

            fp = photo.open()
            img = Image.open(BytesIO(fp.data))
            img.save(img_path, 'png')

            row = [species, obs.id, n_photos, np.nan]
            metadata.loc[photo.id] = row

            images_stored += 1
            if images_stored >= max_images:
                break

        metadata.species = metadata.species.astype('category')
        metadata.label = metadata.species.cat.codes
        metadata.to_csv(metadata_path)

        if len(metadata) >= max_images:
            return True

    return True


class InatDataset(Dataset):

    def __init__(self, data_dir: os.PathLike, metadata: pd.DataFrame):
        super().__init__()
        self.data_dir = data_dir
        self.metadata = metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index) -> T_co:
        row = self.metadata.loc[index]
        y = row.label
        img = Image.open(os.path.join(self.data_dir, row.species, f"{row.photo_id}.png"))
        x = pil_to_tensor(img)
        return x, y


class InatDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: os.PathLike, species=None):
        super().__init__()
        self.data_dir = data_dir
        InatDataModule.validate_data_dir(data_dir)
        self.metadata = self.read_metadata(species)

    @staticmethod
    def validate_data_dir(data_dir):
        if not os.path.isdir(data_dir):
            raise ValueError(f"data_dir={data_dir}: No such directory found.")
        if not os.path.exists(os.path.join(data_dir, "metadata.csv")):
            raise ValueError(f"No metadata.csv file found in {data_dir}")
        return True

    def read_metadata(self, species=None):
        if species is None:
            species = []
        if isinstance(species, str):
            species = [species]

        metadata = pd.read_csv(os.path.join(self.data_dir, "metadata.csv"))
        metadata.reset_index(inplace=True)
        metadata.drop(columns='index', inplace=True)
        metadata.species = metadata.species.astype('category')

        if len(species):
            metadata = metadata[metadata.species.isin(species)]

        return metadata

    def setup(self, stage: Optional[str] = None) -> None:

        # TODO: fill other classes somewhere else
        # balance classes
        groupby = self.metadata.groupby('label')
        groupby.apply(lambda x: x.sample(groupby.size().min()).reset_index(drop=True))

        if stage == "fit" or stage is None:
            pass
        pass


from typing import Optional
import os
import pandas as pd
import pyinaturalist as pin
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from collections import Counter

from torchvision.datasets import ImageFolder
from torchvision import transforms

from CitizenUAV.transforms import *


def download_data(species: str, output_dir: os.PathLike, max_images: Optional[int] = None,
                  min_year: Optional[int] = 2010):
    """
    Download inaturalist image data for a certain species.
    :param species: species to collect data for
    :param output_dir: output directory
    :param max_images: maximum number of images to download
    :param min_year: year of the earliest observations to collect
    :return pd.DataFrame: collected metadata
    """
    quality = "research"

    # create directory
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # load or create metadata DataFrame
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)
        metadata.reset_index(drop=True, inplace=True)
        metadata.set_index('photo_id', inplace=True)
    else:
        metadata = pd.DataFrame(columns=['species', 'obs_id', 'n_photos', 'label'])
        metadata.index.name = 'photo_id'

    # collect data from inaturalist
    response = pin.get_observations(
        taxon_name=species,
        quality_grade=quality,
        photos=True,
        page='all',
        year=range(int(min_year), 2024)
    )
    obss = pin.Observations.from_json_list(response)

    # break process if no data was found
    if not len(obss):
        print(f"No observations found for species {species}")
        return False

    # create species directory if it doesn't exist
    spec_dir = os.path.join(output_dir, species)
    if not os.path.isdir(spec_dir):
        os.makedirs(spec_dir)

    images_stored = 0
    # iterate over observations
    p_bar = tqdm(obss)
    p_bar.set_description(f"Downloading data for species {species} ...")
    for obs in p_bar:

        n_photos = len(obs.photos)

        # iterate over images in observation
        for i, photo in enumerate(obs.photos):

            # set file name with photo id
            filename = f'{photo.id}.png'
            img_path = os.path.join(spec_dir, filename)

            # create entry in metadata
            row = [species, obs.id, n_photos, np.nan]
            if len(metadata.columns) > len(row):
                # deal with existing extended metadata
                row += [np.nan] * (len(metadata.columns) - len(row))
            metadata.loc[photo.id] = row

            # skip photo, if already downloaded
            if os.path.exists(img_path):
                continue

            # download image
            fp = photo.open()
            img = Image.open(BytesIO(fp.data))
            img.save(img_path, 'png')

            # count image and stop if enough images have been downloaded
            images_stored += 1
            if max_images is not None and images_stored >= max_images:
                break

        # format class label column in metadata
        metadata.species = metadata.species.astype('category')
        metadata.label = metadata.species.cat.codes
        metadata.to_csv(metadata_path)

        # stop whole procedure if enough images have been downloaded
        if max_images is not None and len(metadata[metadata.species == species]) >= max_images:
            return metadata

    return metadata


def extend_metadata(data_dir, consider_augmented=False):
    csv_path = os.path.join(data_dir, 'metadata.csv')
    ds = ImageFolder(data_dir, transform=transforms.ToTensor())
    metadata = pd.read_csv(csv_path)
    metadata.set_index('photo_id', inplace=True)

    idx = range(len(ds))
    if not consider_augmented:
        # skip augmented images
        idx = [i for i in idx if "_" not in os.path.basename(ds.samples[i][0])]

    max_vals = pd.Series(index=metadata.index, dtype='float32')
    min_vals = pd.Series(index=metadata.index, dtype='float32')
    mean_vals = pd.Series(index=metadata.index, dtype='float32')
    paths = pd.Series(index=metadata.index, dtype=str)
    heights = pd.Series(index=metadata.index, dtype='int32')
    widths = pd.Series(index=metadata.index, dtype='int32')
    contrasts = pd.Series(index=metadata.index, dtype='float32')
    saturations = pd.Series(index=metadata.index, dtype='float32')
    broken = pd.Series(index=metadata.index, dtype=bool)
    broken[:] = False

    for i in tqdm(idx):
        path, _ = ds.samples[i]
        filename = os.path.splitext(os.path.basename(path))[0]
        pid = int(filename)

        try:
            img, cls_idx = ds[i]
        except OSError:
            # skip and mark broken files
            broken[pid] = True
            print(f"Skipping broken file with id {pid} ...")
            continue

        cls = ds.classes[cls_idx]

        try:
            row = metadata.loc[pid]
        except KeyError:
            new_row = [cls, np.nan, np.nan, cls_idx]
            if len(metadata.columns) > len(new_row):
                new_row += [np.nan] * (len(metadata.columns) - len(new_row))
            metadata.loc[pid] = new_row
            row = metadata.loc[pid]
        if cls != row.species:
            raise ValueError(f"Classes {cls} and {metadata['species']} do not match for image {pid}!")

        max_val = float(torch.max(img).numpy())
        max_vals[pid] = max_val
        min_val = float(torch.min(img).numpy())
        min_vals[pid] = min_val
        mean_vals[pid] = float(torch.mean(img).numpy())
        channels, heights[pid], widths[pid], = img.size()
        contrast = max_val - min_val
        contrasts[pid] = contrast
        saturations[pid] = contrast / max_val
        paths[pid] = path

    # NaN entries might have been created. Replace them with the default value.
    broken.fillna(False, inplace=True)

    metadata['max_val'] = max_vals
    metadata['min_val'] = min_vals
    metadata['mean_val'] = mean_vals
    metadata['path'] = paths
    metadata['height'] = heights
    metadata['width'] = widths
    metadata['contrast'] = contrasts
    metadata['saturation'] = saturations
    metadata['broken'] = broken

    metadata.to_csv(csv_path)

    return metadata


def offline_augmentation(data_dir: os.PathLike, target_n):
    """
    Perform offline augmentation on present images.
    :param data_dir: The directory where the data lies.
    :param target_n: Target number of samples per class.
    :return: True, if the procedure succeeded.
    """
    # create dataset
    ds = ImageFolder(data_dir, transform=transforms.Compose([transforms.ToTensor(), QuadCrop()]))

    # count samples per class
    n_samples = dict(Counter(ds.targets))

    # define repertoire of augmentation techniques
    techniques = [RandomBrightness(), RandomContrast(), RandomSaturation()]

    # postprocessing including random flips
    do_anyway_after = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToPILImage()]

    # iterate over classes
    for cls, n in n_samples.items():
        pbar = tqdm(range(target_n - n))
        pbar.set_description(f"Augmenting for {ds.classes[cls]}")

        # only consider indices of current class for augmentation sampling
        cls_idx = [i for i in range(len(ds)) if ds.targets[i] == cls]

        for _ in pbar:
            # sample image to be modified
            rand_idx = np.random.choice(cls_idx)
            img, y = ds[rand_idx]

            # Apply each technique with a probability of 0.5 (don't confuse with internal random parameters).
            random_apply = transforms.RandomApply(techniques)

            # compose and apply modification pipeline
            transform = transforms.Compose([random_apply] + do_anyway_after)
            augmented = transform(img)

            # determine new file name
            orig_filepath = ds.samples[rand_idx][0]
            filepath = os.path.splitext(orig_filepath)[0]
            n_copies = 1
            # make sure the filename doesn't exist yet
            while os.path.exists(f"{filepath}_{n_copies}.png"):
                n_copies += 1
            filepath = f"{filepath}_{n_copies}.png"
            # save new image
            augmented.save(filepath, 'png')

    return True

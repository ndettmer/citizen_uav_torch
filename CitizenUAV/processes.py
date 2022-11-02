from typing import Optional
import os
import pandas as pd
import pyinaturalist as pin
from torch.utils.data import Subset
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from collections import Counter

from torchvision.datasets import ImageFolder
from torchvision import transforms

from CitizenUAV.models import InatRegressor
from CitizenUAV.transforms import *
from CitizenUAV.data import InatDistDataset, InatDistDataModule


def download_data(species: str, output_dir: os.PathLike, max_images: Optional[int] = None,
                  min_year: Optional[int] = 2010, max_year: Optional[int] = 2024):
    """
    Download inaturalist image data for a certain species.
    :param species: species to collect data for
    :param output_dir: output directory
    :param max_images: maximum number of images to download
    :param min_year: year of the earliest observations to collect
    :param max_year: year of the latest observations to collect
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

    if max_images is not None and len(metadata[metadata.species == species]) >= max_images:
        return metadata

    # collect data from inaturalist
    response = pin.get_observations(
        taxon_name=species,
        quality_grade=quality,
        photos=True,
        page='all',
        year=range(int(min_year), int(max_year))
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

        n_changes = 0

        # iterate over images in observation
        for i, photo in enumerate(obs.photos):

            # set file name with photo id
            filename = f'{photo.id}.png'
            img_path = os.path.join(spec_dir, filename)

            # skip photo, if already downloaded and information collected
            if os.path.exists(img_path) and photo.id in metadata.index:
                continue

            # download image
            fp = photo.open()
            img = Image.open(BytesIO(fp.data))
            img.save(img_path, 'png')

            # create entry in metadata
            row = [species, obs.id, n_photos, np.nan]
            if len(metadata.columns) > len(row):
                # deal with existing extended metadata
                row += [np.nan] * (len(metadata.columns) - len(row))
            metadata.loc[photo.id] = row
            n_changes += 1

            # count image and stop if enough images have been downloaded
            images_stored += 1
            if max_images is not None and images_stored >= max_images:
                break

        # format class label column in metadata
        if n_changes > 0:
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

        # cls = ds.classes[cls_idx]

        # try:
        #    row = metadata.loc[pid]
        # except KeyError:
        #    new_row = [cls, np.nan, np.nan, cls_idx]
        #    if len(metadata.columns) > len(new_row):
        #        new_row += [np.nan] * (len(metadata.columns) - len(new_row))
        #    metadata.loc[pid] = new_row
        #    row = metadata.loc[pid]
        # if cls != row.species:
        #    raise ValueError(f"Classes {cls} and {metadata['species']} do not match for image {pid}!")

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


def extend_dist_metadata(data_dir, consider_augmented=False):
    csv_path = os.path.join(data_dir, 'distances.csv')
    ds = InatDistDataset(data_dir, transforms.ToTensor())
    metadata = pd.read_csv(csv_path)

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

    p_bar = tqdm(range(len(ds)))
    p_bar.set_description("Extending metadata for dataset ...")
    for i in p_bar:

        filename, distance = ds.samples[i]
        path = os.path.join(data_dir, filename)

        try:
            img, t = ds[i]
        except OSError:
            # skip and mark broken files
            broken[i] = True
            print(f"Skipping broken file with index {i} ...")
            continue

        max_val = float(torch.max(img).numpy())
        max_vals[i] = max_val
        min_val = float(torch.min(img).numpy())
        min_vals[i] = min_val
        mean_vals[i] = float(torch.mean(img).numpy())
        channels, heights[i], widths[i], = img.size()
        contrast = max_val - min_val
        contrasts[i] = contrast
        saturations[i] = contrast / max_val
        paths[i] = path

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


def check_images(data_dir):
    csv_path = os.path.join(data_dir, "metadata.csv")
    metadata = pd.read_csv(csv_path)
    metadata.set_index('photo_id', inplace=True)
    ds = ImageFolder(data_dir, transform=transforms.ToTensor())

    image_okay = pd.Series(index=metadata.index, dtype=bool)

    p_bar = tqdm(range(len(ds)))
    p_bar.set_description(f"Checking images in {data_dir} ...")
    for i in p_bar:
        path, _ = ds.samples[i]
        filename = os.path.splitext(os.path.basename(path))[0]
        pid = int(filename)

        try:
            img, t = ds[i]
            image_okay[pid] = True
        except OSError:
            image_okay[pid] = False

    metadata['image_okay'] = image_okay
    metadata.to_csv(csv_path)


def get_pid_from_path(path):
    # TODO: move to utils
    filename = os.path.splitext(os.path.basename(path))[0]
    return int(filename)


def predict_distances(data_dir, model_path, train_min, train_max, img_size=256, species: list[str] = None,
                      batch_size: int = 1, gpu: bool = True):
    # TODO: test
    csv_path = os.path.join(data_dir, "metadata.csv")
    ds = ImageFolder(data_dir, transform=transforms.Compose([
        transforms.ToTensor(), QuadCrop(), transforms.Resize(img_size), Log10()
    ]))

    idx = range(len(ds))

    if species:
        idx = [i for i in idx if ds.classes[ds.targets[i]] in species]

    metadata = pd.read_csv(csv_path)
    metadata.set_index('photo_id', inplace=True)

    if 'image_okay' not in metadata.columns:
        del metadata
        check_images(data_dir)
        metadata = pd.read_csv(csv_path)
        metadata.set_index('photo_id', inplace=True)

    # filter for loadable images
    metadata = metadata[metadata.image_okay]
    idx = [i for i in idx if get_pid_from_path(ds.samples[i][0]) in metadata.index]

    distances = pd.Series(index=metadata.index, dtype='float32')

    model = InatRegressor.load_from_checkpoint(model_path)
    model.eval()

    if gpu:
        model.cuda()

    idx = np.array(idx)
    p_bar = tqdm(range(len(idx) // batch_size))
    p_bar.set_description(f"Predicting distances ...")
    for batch_no in p_bar:

        # build batch
        pids = []
        images = []
        for i in idx[batch_no * batch_size:(batch_no+1) * batch_size]:
            path, _ = ds.samples[i]
            filename = os.path.splitext(os.path.basename(path))[0]

            pid = int(filename)
            pids.append(pid)

            img, _ = ds[i]
            images.append(img)

        batch = torch.stack(images, dim=0)
        if gpu:
            batch.cuda()

        # make predictions
        with torch.no_grad():
            preds = model(batch)

        # predict raw/real distances
        raw_distances = preds * (train_max - train_min) + train_min
        raw_distances = raw_distances.squeeze(1).numpy()

        # write to series
        for pid, dist in zip(pids, raw_distances):
            distances[pid] = dist

        # TODO: remove debug
        if batch_no >= 10:
            break

    # TODO: make security saves
    metadata['distance'] = distances
    metadata.to_csv(csv_path)
    return metadata

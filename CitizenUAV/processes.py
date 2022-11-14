from typing import Optional
import pyinaturalist as pin
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from collections import Counter

from torchvision.datasets import ImageFolder
from torchvision import transforms

from CitizenUAV.models import InatRegressor
from CitizenUAV.transforms import *
from CitizenUAV.data import InatDistDataset
from CitizenUAV.utils import *


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
            metadata.loc[str(photo.id)] = row
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
        idx = [i for i in idx if "augmented" not in os.path.basename(ds.samples[i][0])]

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
        pid = get_pid_from_path(path)

        try:
            img, cls_idx = ds[i]
        except OSError:
            # skip and mark broken files
            broken[pid] = True
            print(f"Skipping broken file with id {pid}.")
            continue

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
    """
    Extend metadata of distance regression dataset.
    :param data_dir: Path to the image dataset.
    :param consider_augmented: If true, also consider images created in an augmentation process
        (currently not implemented).
    :return pd.DataFrame: Extended metadata.
    """
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

    metadata.to_csv(csv_path, index=False)

    return metadata


def offline_augmentation_regression_data(data_dir: os.PathLike, target_n, debug: bool = False):
    """
    Perform offline augmentation on distance-labelled image dataset.
    :param data_dir: The directory where the data lies. (There has to be a single subdirectory containing the data.
        E.g.: data_dir = '/some/path/', so the images should be in '/some/path/subdir/img.png' That is because
        torchvision.datasets.ImageFolder is used for dataset management.)
    :param target_n: Target number of samples per class.
    :param debug: In debug mode no file changes or new files are saved.
    :return: True, if the procedure succeeded.
    """
    # create dataset
    ds = ImageFolder(data_dir, transform=transforms.Compose([transforms.ToTensor(), QuadCrop()]))

    idx = range(len(ds))

    # filter augmented samples
    idx = [i for i in idx if 'augmented' not in os.path.basename(ds.samples[i][0])]

    # define augmentation pipeline
    transform = transforms.Compose([
        RandomBrightness(),
        RandomContrast(),
        RandomSaturation(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        Clamp(),
        transforms.ToPILImage()
    ])

    # iterate over classes
    pbar = tqdm(range(target_n - len(ds)))
    pbar.set_description(f"Augmenting for distance regression training data in directory: {data_dir}")

    # load distances.csv
    subdir = next(os.walk(data_dir))[1][0]
    dist_csv_path = os.path.join(data_dir, subdir, 'distances.csv')
    dist_df = pd.read_csv(dist_csv_path)

    for _ in pbar:
        # sample image to be modified
        rand_idx = np.random.choice(idx)
        img, y = ds[rand_idx]

        if img.min() < 0 or img.max() > 1:
            raise ValueError("Image is not normalized to [0;1]!")

        # apply augmentation pipeline
        augmented = transform(img)

        # determine new file name
        orig_filepath = ds.samples[rand_idx][0]

        # If the original file has no entry in the dataframe, something went wrong.
        # Therefore, skip storing the new image.
        # TODO: here is a dependency to extend_metadata
        orig_slice = dist_df[dist_df.path == orig_filepath]
        if not len(orig_slice):
            continue

        # make sure the filename doesn't exist yet
        filepath = os.path.splitext(orig_filepath)[0]
        n_copies = 1
        while os.path.exists(f"{filepath}_augmented_{n_copies}.png"):
            n_copies += 1
        filepath = f"{filepath}_augmented_{n_copies}.png"

        # Add entry to distances.csv
        row = pd.Series(index=dist_df.columns, dtype=float)
        row.Image = os.path.basename(filepath)
        row.Distance = orig_slice.iloc[0].Distance
        if 'path' in dist_df.columns:
            row.path = filepath
        if 'broken' in dist_df.columns:
            row.broken = False
        dist_df.loc[len(dist_df)] = row
        if not debug:
            dist_df.to_csv(dist_csv_path, index=False)

        # save new image
        if not debug:
            augmented.save(filepath, 'png')

    return True


def offline_augmentation_classification_data(data_dir: os.PathLike, target_n, subdirs: list[str] = None,
                                             min_distance: float = None, debug: bool = False):
    """
    Perform offline augmentation on species-labelled image dataset.
    :param data_dir: The directory where the data lies.
    :param target_n: Target number of samples per class.
    :param subdirs: Sub-directories to consider (handled as targets in the ImageFolder class).
    :param min_distance: Minimum distance to be considered for augmentation source.
    :param debug: In debug mode no file changes or new files are saved.
    :return: True, if the procedure succeeded.
    """
    # create dataset
    ds = ImageFolder(data_dir, transform=transforms.Compose([transforms.ToTensor(), QuadCrop()]))
    idx = range(len(ds))

    # filter augmented samples
    idx = [i for i in idx if 'augmented' not in os.path.basename(ds.samples[i][0])]

    # only consider samples of specified classes
    if subdirs:
        idx = [i for i in idx if ds.classes[ds.targets[i]] in subdirs]

    metadata = read_inat_metadata(data_dir)
    if min_distance:
        # load metadata file and filter samples by distance
        min_dist_pids = metadata[metadata.distance >= min_distance].index
        idx = [i for i in idx if get_pid_from_path(ds.samples[i][0]) in min_dist_pids]

        # count samples per class
        dist_cleaned_targets = [ds.targets[i] for i in idx]
        n_samples = dict(Counter(dist_cleaned_targets))
    else:
        n_samples = dict(Counter(ds.targets))

    # define augmentation pipeline
    transform = transforms.Compose([
        RandomBrightness(),
        RandomContrast(),
        RandomSaturation(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        Clamp(),
        transforms.ToPILImage()
    ])

    # iterate over classes
    for cls, n in n_samples.items():

        # Only consider indices of current class for augmentation sampling.
        cls_idx = [i for i in idx if ds.targets[i] == cls]

        # If no indices are left after filtering, skip this class.
        if not len(cls_idx):
            continue

        # The number of augmented images to be created is the difference between the target number of samples per class
        # and the actual number of samples for that class.
        pbar = tqdm(range(target_n - n))
        pbar.set_description(f"Augmenting for class: {ds.classes[cls]}")

        for _ in pbar:
            # sample image to be modified
            rand_idx = np.random.choice(cls_idx)
            img, y = ds[rand_idx]

            if img.min() < 0 or img.max() > 1:
                raise ValueError("Image is not normalized to [0;1]!")

            # apply augmentation pipeline
            augmented = transform(img)

            # determine new file name
            orig_filepath = ds.samples[rand_idx][0]
            filepath = os.path.splitext(orig_filepath)[0]
            n_copies = 1
            # make sure the filename doesn't exist yet
            while os.path.exists(f"{filepath}_augmented_{n_copies}.png"):
                n_copies += 1
            filepath = f"{filepath}_augmented_{n_copies}.png"

            if not debug:
                row = metadata.loc[get_pid_from_path(orig_filepath)].copy()
                metadata.loc[get_pid_from_path(filepath)] = row
                metadata.to_csv(os.path.join(data_dir, 'metadata.csv'))

            # save new image
            if not debug:
                augmented.save(filepath, 'png')

    return True


def check_image_files(data_dir):
    """
    Check image files in directory for being loadable.
    :param data_dir: Directory of the image data set
    """
    csv_path = os.path.join(data_dir, "metadata.csv")
    metadata = read_inat_metadata(data_dir)
    ds = ImageFolder(data_dir, transform=transforms.ToTensor())

    image_okay = pd.Series(index=metadata.index, dtype=bool)

    p_bar = tqdm(range(len(ds)))
    p_bar.set_description(f"Checking image files in {data_dir} ...")
    for i in p_bar:
        path, _ = ds.samples[i]
        filename = os.path.splitext(os.path.basename(path))[0]
        pid = int(filename)

        try:
            img, t = ds[i]
            image_okay.loc[pid] = True
        except (OSError, UnidentifiedImageError):
            image_okay[pid] = False

    metadata['image_okay'] = image_okay
    metadata.to_csv(csv_path)


def predict_distances(data_dir, model_path, train_min, train_max, img_size=256, species: list[str] = None,
                      batch_size: int = 1, gpu: bool = True, overwrite: bool = False,
                      debug: bool = False) -> pd.DataFrame:
    """
    Predicts acquisition distances for an image dataset.
    :param data_dir: Directory holding the image data
    :param model_path: Path to the trained model (pytorch checkpoint).
    :param train_min: Minimum real distance in the training dataset of the model.
    :param train_max: Maximum real distance in the training dataset of the model.
    :param img_size: Input image size of the model.
    :param species: List of subdirectories to be considered
        (in INaturalist datasets they are named by the plant species)
    :param batch_size: Batch size
    :param gpu: If true, use GPU for model inference
    :param overwrite: If true, overwrite previously assigned distances
    :param debug: No files distances are saved in debug mode.
    :return pd.DataFrame: Updated metadata containing distance predictions in column "distance"
    """

    csv_path = os.path.join(data_dir, "metadata.csv")
    ds = ImageFolder(data_dir, transform=transforms.Compose([
        transforms.ToTensor(), QuadCrop(), transforms.Resize(img_size), Log10()
    ]))

    idx = range(len(ds))

    if species:
        idx = [i for i in idx if ds.classes[ds.targets[i]] in species]

    metadata = pd.read_csv(csv_path)
    metadata.photo_id = metadata.photo_id.astype(str)
    metadata.set_index('photo_id', inplace=True)

    if 'image_okay' not in metadata.columns:
        del metadata
        check_image_files(data_dir)
        metadata = read_inat_metadata(data_dir)

    if not overwrite:
        # only consider samples that don't have a distance assigned yet
        na_pids = metadata[metadata.distance.isna()].index
        idx = [i for i in idx if get_pid_from_path(ds.samples[i][0]) in na_pids]
        del na_pids

    # filter for loadable images
    okay_pids = metadata[metadata.image_okay].index
    idx = [i for i in idx if get_pid_from_path(ds.samples[i][0]) in okay_pids]
    del okay_pids

    # if no samples are left after filtering, end process
    if not len(idx):
        return metadata

    if 'distance' in metadata.columns:
        distances = metadata.distance
    else:
        distances = pd.Series(index=metadata.index, dtype='float32')

    model = InatRegressor.load_from_checkpoint(model_path)
    model.eval()

    if gpu:
        model.cuda()

    if batch_size > len(idx):
        batch_size = len(idx)

    idx = np.array(idx)
    p_bar = tqdm(range(len(idx) // batch_size))
    p_bar.set_description(f"Predicting distances ...")
    for batch_no in p_bar:

        # build batch
        pids = []
        images = []
        for i in idx[batch_no * batch_size:(batch_no + 1) * batch_size]:
            path, _ = ds.samples[i]
            filename = os.path.splitext(os.path.basename(path))[0]

            pid = filename
            pids.append(pid)

            img, _ = ds[i]
            img = img.float()
            images.append(img)

        batch = torch.stack(images, dim=0)
        if gpu:
            batch = batch.cuda()

        # make predictions
        with torch.no_grad():
            preds = model(batch)

        # predict raw/real distances
        raw_distances = preds * (train_max - train_min) + train_min
        raw_distances = raw_distances.squeeze(1).cpu().numpy()

        # write to series
        for pid, dist in zip(pids, raw_distances):
            distances[pid] = dist

        # make intermediate saves
        if not batch_no % 5:
            metadata['distance'] = distances
            if not debug:
                metadata.to_csv(csv_path)

    metadata['distance'] = distances
    if not debug:
        metadata.to_csv(csv_path)
    return metadata


def create_gim_metadata(data_dir: os.PathLike, classes: list = None, debug: bool = False):

    metadata = pd.DataFrame(columns=["photo_id", "species", "distance", "obs_id", "n_photos", "label", "image_okay"])
    metadata.set_index('photo_id', inplace=True)

    ds = ImageFolder(str(data_dir))
    idx = range(len(ds))

    if classes:
        idx = [i for i in idx if ds.classes[ds[i][1]] in classes]

    p_bar = tqdm(idx)
    p_bar.set_description(f"Creating metadata for image dataset in {data_dir}.")
    for i in p_bar:
        img, _ = ds[i]

        path, t = ds.samples[i]
        row = [ds.classes[t], 150., np.nan, 1, np.nan, True]

        photo_id, _ = os.path.splitext(os.path.basename(path))
        metadata.loc[photo_id] = row

    metadata.species = metadata.species.astype('category')
    metadata.label = metadata.species.cat.codes

    if not debug:
        metadata.to_csv(os.path.join(data_dir, "metadata.csv"))

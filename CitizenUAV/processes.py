import pyinaturalist as pin
import json

import torch.cuda
from PIL import UnidentifiedImageError
from io import BytesIO
from scipy.stats import zscore

from captum.attr import IntegratedGradients, Occlusion, GuidedGradCam
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from torch import optim

from CitizenUAV.losses import ContentLoss, StyleLoss
from CitizenUAV.math_utils import get_area_around_center, get_center_of_bb
from CitizenUAV.models import *
from CitizenUAV.data import *
from CitizenUAV.io_utils import *


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
    # By this label-security is ensured.
    quality = "research"

    # create directory
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # load or create metadata DataFrame
    metadata_path = os.path.join(output_dir, species, 'metadata.csv')
    metadata_backup_path = os.path.join(output_dir, species, 'metadata_backup.csv')
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
    spec_img_dir = os.path.join(output_dir, species)
    if not os.path.isdir(spec_img_dir):
        os.makedirs(spec_img_dir)

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
            img_path = os.path.join(spec_img_dir, filename)

            # check if data already exists
            image_exists = os.path.exists(img_path)
            if image_exists:
                try:
                    test_img = Image.open(img_path)
                except (OSError, UnidentifiedImageError):
                    image_exists = False
                finally:
                    del test_img
            metadata_exists = photo.id in metadata.index

            # skip photo, if already downloaded and information collected
            if image_exists and metadata_exists:
                continue

            # If image doesn't exist or is broken, download image.
            if not image_exists:
                fp = photo.open()
                img = Image.open(BytesIO(fp.data))
                try:
                    img.save(img_path, 'png')
                except KeyboardInterrupt:
                    img.save(img_path, 'png')
                    sys.exit()

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
            try:
                metadata.to_csv(metadata_path)
                metadata.to_csv(metadata_backup_path)
            except KeyboardInterrupt:
                metadata.to_csv(metadata_path)
                metadata.to_csv(metadata_backup_path)
                sys.exit()

        # stop whole procedure if enough images have been downloaded
        if max_images is not None and len(metadata[metadata.species == species]) >= max_images:
            return metadata

    return metadata


def extend_metadata(data_dir, consider_augmented=False):
    ds = ImageFolder(data_dir, transform=transforms.ToTensor())
    metadata = read_split_inat_metadata(data_dir)

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

    pbar = tqdm(idx)
    pbar.set_description(f"Extending metadata in {data_dir}")
    for i in pbar:
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

    store_split_inat_metadata(metadata, data_dir)

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
    Perform offline augmentation on distance-labeled image dataset.
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


def offline_augmentation_classification_data(data_dir: Union[str, Path], target_n, subdirs: list[str] = None,
                                             no_metadata: bool = False, min_distance: float = None, debug: bool = False):
    """
    Perform offline augmentation on species-labeled image dataset.
    :param data_dir: The directory where the data lies.
    :param target_n: Target number of samples per class.
    :param subdirs: Sub-directories to consider (handled as targets in the ImageFolder class).
    :param no_metadata: Ignore metadata.csv (e.g. if it doesn't exist).
    :param min_distance: Minimum distance to be considered for augmentation source.
    :param debug: In debug mode no file changes or new files are saved.
    :return: True, if the procedure succeeded.
    """
    if no_metadata and min_distance is not None:
        raise ValueError("metadata.csv is needed for distance filtering.")

    # create dataset
    ds = ImageFolder(data_dir, transform=transforms.Compose([transforms.ToTensor(), QuadCrop()]))
    idx = range(len(ds))

    # only consider samples of specified classes
    if subdirs:
        idx = [i for i in idx if ds.classes[ds.targets[i]] in subdirs]

    if not no_metadata:
        metadata = read_split_inat_metadata(data_dir)
    if min_distance:
        # load metadata file and filter samples by distance
        min_dist_pids = metadata[metadata.distance >= min_distance].index
        idx = [i for i in idx if get_pid_from_path(ds.samples[i][0]) in min_dist_pids]

        # count samples per class
        dist_cleaned_targets = [ds.targets[i] for i in idx]
        n_samples = dict(Counter(dist_cleaned_targets))
    else:
        n_samples = dict(Counter(ds.targets))

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
                if not no_metadata:
                    # store updated metadata
                    row = metadata.loc[get_pid_from_path(orig_filepath)].copy()
                    metadata.loc[get_pid_from_path(filepath)] = row
                    store_split_inat_metadata(metadata, data_dir)

                # save new image
                augmented.save(filepath, 'png')

    return True


def remove_augmented_classification_images(data_dir: Union[str, Path], subdirs: Optional[list[str]] = None):
    metadata = read_split_inat_metadata(data_dir, subdirs)
    ds = ImageFolder(data_dir)
    idx = range(len(ds))
    if subdirs is not None:
        idx = [i for i in idx if ds.classes[ds.targets[i]] in subdirs]

    pids = []
    pbar = tqdm(idx)
    pbar.set_description("Removing augmented images")
    for i in pbar:
        path, _ = ds.samples[i]
        if "augmented" in os.path.basename(path):
            pids.append(get_pid_from_path(path))
            if os.path.exists(path):
                os.remove(path)

    metadata.drop(index=pids, inplace=True)
    store_split_inat_metadata(metadata, data_dir)


def check_image_files(data_dir):
    """
    Check image files in directory for being loadable.
    :param data_dir: Directory of the image data set
    """
    metadata = read_split_inat_metadata(data_dir)
    ds = ImageFolder(data_dir, transform=transforms.ToTensor())

    image_okay = pd.Series(index=metadata.index, dtype=bool)

    p_bar = tqdm(range(len(ds)))
    p_bar.set_description(f"Checking image files in {data_dir} ...")
    for i in p_bar:
        path, _ = ds.samples[i]
        filename = os.path.splitext(os.path.basename(path))[0]
        pid = filename

        try:
            img, t = ds[i]
            image_okay.loc[pid] = True
        except (OSError, UnidentifiedImageError):
            image_okay[pid] = False

    metadata['image_okay'] = image_okay
    store_split_inat_metadata(metadata, data_dir)


def predict_distances(data_dir, model_path, train_min, train_max, img_size=256, species: Optional[list[str]] = None,
                      batch_size: int = 1, gpu: Optional[bool] = None, overwrite: bool = False,
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

    ds = ImageFolder(data_dir, transform=transforms.Compose([
        transforms.ToTensor(), QuadCrop(), transforms.Resize(img_size), Log10()
    ]))

    idx = range(len(ds))

    if species:
        idx = [i for i in idx if ds.classes[ds.targets[i]] in species]

    metadata = read_split_inat_metadata(data_dir, species)

    if 'image_okay' not in metadata.columns:
        del metadata
        check_image_files(data_dir)
        metadata = read_split_inat_metadata(data_dir, species)

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

    if gpu is None:
        gpu = torch.cuda.is_available()

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
            distances.loc[pid] = dist

        # make intermediate saves
        if not batch_no % 5:
            metadata['distance'] = distances
            if not debug:
                store_split_inat_metadata(metadata, data_dir)

    metadata['distance'] = distances
    if not debug:
        store_split_inat_metadata(metadata, data_dir)
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


def predict_inat(model_path: Union[str, Path], data_dir: Union[str, Path], result_dir: Union[str, Path],
                 img_size: int = 128,
                 min_distance: float = None, gpu: Optional[bool] = None, batch_size: int = 1, normalize: bool = False,
                 model_class: str = 'InatSequentialClassifier'):
    dm = InatDataModule(data_dir, img_size=img_size, normalize=normalize, min_distance=min_distance, split=(0, 0, 1),
                        batch_size=batch_size, return_path=True)
    dm.setup()
    model = eval(model_class).load_from_checkpoint(model_path)
    model.eval()

    if gpu is None:
        gpu = torch.cuda.is_available()

    if gpu:
        model.cuda()

    columns = ['pid', 'target', 'target_text', 'prediction', 'prediction_text', 'path']
    for inat_class in dm.ds.classes:
        columns.append(f"{inat_class}_prob")
    pred_df = pd.DataFrame(columns=columns)

    data_iter = iter(dm.test_dataloader())
    for batch, ts, paths in tqdm(data_iter):
        if gpu:
            batch = batch.cuda()

        with torch.no_grad():
            y_hat = model(batch)
        preds = torch.argmax(y_hat, dim=1)

        ts = ts.numpy()
        preds = preds.detach().cpu().numpy()
        y_hat = np.around(y_hat.detach().cpu().numpy(), 4)

        for path, t, pred, probs in zip(paths, ts, preds, y_hat):
            row = pd.Series(index=pred_df.columns, dtype=object)

            row['pid'] = get_pid_from_path(path)
            row['target'] = int(t)
            row['target_text'] = dm.ds.classes[int(t)]
            row['prediction'] = int(pred)
            row['prediction_text'] = dm.ds.classes[int(pred)]
            row['path'] = path

            for i, prob in enumerate(probs):
                row[f'{dm.ds.classes[i]}_prob'] = float(prob)

            pred_df.loc[len(pred_df)] = row

    pred_df['model_path'] = model_path

    model_version = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    result_dir = os.path.join(result_dir, 'predictions', 'inat', model_version)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_filename = f"{datetime.now().strftime('%y-%m-%d_%H-%M')}_predictions.csv"
    pred_df.to_csv(os.path.join(result_dir, result_filename), index=False)


def predict_geotiff(model_path: Union[str | os.PathLike], dataset_path: Union[str | os.PathLike],
                    result_dir: Optional[Union[str | os.PathLike]] = None,
                    window_size: int = 128, stride: int = 1, gpu: bool = False, batch_size: int = 1,
                    normalize: bool = False, means: Optional[tuple] = None, stds: Optional[tuple] = None,
                    probabilities: bool = False, pred_size: int = None, model_class: str = 'InatSequentialClassifier',
                    debug: bool = False, **kwargs) -> tuple[np.ndarray, str]:
    """
    Predict pixel-wise classes for a GeoTiff raster dataset with a given trained model using a moving window approach.
    :param model_path: Path to the model checkpoint.
    :param dataset_path: Path to the GeoTiff raster dataset.
    :param result_dir: Storage directory for the prediction results.
    :param window_size: Size of the moving window.
    :param stride: Step size of the moving window algorithm.
    :param gpu: If true, use GPU. Will be ignored, if CUDA is not available.
    :param batch_size: Batch size for the predictions.
    :param normalize: Normalize dataset.
    :param debug: If true, do some assertions and logging.
    :param means: Normalization channel means. Note that they should be scaled to [0,255]!
    :param stds: Normalization channel stds. Note that they should be scaled to [0,255]!
    :param probabilities: Add confidences and not 1-hot predictions.
    :param pred_size: Apply the prediction only to an area around the center pixel of the bounding box
    :param model_class: Choose a model class from models.py.
    :return: Resulting label map containing the final predictions and the path to the stored label map.
    """

    if result_dir is None:
        result_dir = os.getcwd()
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    version_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    result_path = os.path.join(result_dir, "predictions", dataset_name, version_name,
                               f'{datetime.now().strftime("%y-%m-%d_%H-%M")}_{"probability_map" if probabilities else "label_map"}.npy')
    result_box_path = result_path.replace(".npy", ".csv")

    if not os.path.exists(result_path):
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

    ds = GTiffDataset(dataset_path, window_size=window_size, stride=stride, normalize=normalize, means=means, stds=stds)

    model = eval(model_class).load_from_checkpoint(model_path)
    model.eval()

    if gpu and torch.cuda.is_available():
        model.cuda()

    # count the predictions in here in order to take the class with the maximum votes afterwards for each pixel.
    label_map = np.full((model.n_classes, ds.rds.height, ds.rds.width), 0,
                        dtype=(np.float16 if probabilities else np.uint8))
    box_preds = []
    if debug:
        assert ds.get_mask_bool().shape == label_map.shape[1:]

    p_bar = tqdm(range(len(ds) // batch_size))
    p_bar.set_description(f"Predicting classes in raster dataset")
    for batch_no in p_bar:
        batch_bbs = ds.bbs[batch_no * batch_size:(batch_no + 1) * batch_size]
        batch = torch.stack([ds.get_bb_data(bb) for bb in batch_bbs], dim=0)

        if gpu and torch.cuda.is_available():
            batch = batch.cuda()

        with torch.no_grad():
            y_hat = model(batch)

        preds = torch.argmax(y_hat, dim=1)
        for pred, probs, bb in zip(preds, y_hat, batch_bbs):
            x_min, x_max, y_min, y_max = bb
            conf = float(torch.max(probs).cpu().numpy())
            box_preds.append(BoxPred(
                x_min, x_max, y_min, y_max,
                int(pred.cpu().numpy()),
                ds_path=dataset_path,
                model_path=model_path,
                confidence=conf
            ).dict())

            if pred_size is not None:
                # Calculate prediction bounding box
                x_c, y_c = get_center_of_bb(bb)
                pred_x_min, pred_x_max, pred_y_min, pred_y_max = get_area_around_center(x_c, y_c, pred_size)
            else:
                # Take receptive bounding box as prediction bounding box
                pred_x_min, pred_x_max, pred_y_min, pred_y_max = bb

            if probabilities:
                # add class probability
                label_map[:, pred_x_min:pred_x_max, pred_y_min:pred_y_max] = probs.cpu().numpy().reshape(
                    model.n_classes, 1, 1)
            else:
                # add one-hot prediction
                label_map[pred, pred_x_min:pred_x_max, pred_y_min:pred_y_max] += 1

    p_df = pd.DataFrame(data=box_preds)
    del box_preds
    p_df.to_csv(result_box_path, index=False)

    if probabilities:
        np.save(result_path, label_map)
        return label_map, result_path

    voting_result = np.argmax(label_map, axis=0)
    np.save(result_path, voting_result)
    return voting_result, result_path


def pixel_conf_mat(
        dataset_path: Union[str, Path],
        shape_dir: Union[str, Path],
        train_data_dir: Union[str, Path],
        pred_file: Union[str, Path],
        class_map: Optional[str] = None,
        result_dir: Optional[Union[str, Path]] = None,
        debug: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Creating a pixel-wise confusion matrix for a trained model applied to a GeoTiff dataset.
    :param dataset_path: Path to the raster dataset.
    :param shape_dir: Directory containing the shapefiles that specify the labels of the raster dataset.
    :param train_data_dir: Directory of the training data needed for getting information about the training dataset.
    :param pred_file: Path to the probability map that came out of `predict_geotiff()`.
    :param class_map: JSON formatted mapping from raster classes to training dataset classes.
    :param result_dir: Directory where to store the confusion matrix as a PNG file.
    :param debug: If True, make additional checks.
    :return: The pixel-wise confusion map, a DataFrame containing the pixel-wise predictions and targets, and the
        corresponding F1 score.
    """
    # Load information about training data
    train_ds = ImageFolder(train_data_dir)
    inat_classes = train_ds.classes
    inat_class_to_idx = train_ds.class_to_idx
    del train_ds

    # load evaluation dataset
    window_size = 128
    ds = GTiffDataset(dataset_path, shape_dir=shape_dir, window_size=window_size, stride=window_size)

    # map labels between datasets
    if class_map is None:
        rst_classes_to_inat_classes = {c: c for c in inat_classes}
    else:
        rst_classes_to_inat_classes = json.loads(class_map)
    rst_cls_idx_to_inat_cls_idx = {}
    for rst_cls in ds.classes:
        rst_cls_idx = ds.class_to_idx[rst_cls]
        inat_cls = rst_classes_to_inat_classes[rst_cls]
        inat_cls_idx = inat_class_to_idx[inat_cls]
        rst_cls_idx_to_inat_cls_idx[rst_cls_idx] = inat_cls_idx
    inat_cls_idx_to_rst_cls_idx = {v: k for k, v in rst_cls_idx_to_inat_cls_idx.items()}

    # load predictions and confidences
    prob_map = np.load(pred_file)
    pred_map = np.argmax(prob_map, axis=0)
    del prob_map

    # create target map with inat targets
    target_map = np.full((len(inat_classes), ds.rds.height, ds.rds.width), 0, dtype=bool)

    for inat_class in inat_classes:
        # get inat class index
        inat_cls_idx = inat_class_to_idx[inat_class]
        # get corresponding raster class index
        rst_cls_idx = inat_cls_idx_to_rst_cls_idx[inat_cls_idx]

        # get class target coverage from raster dataset
        cls_mask = ds.get_cls_mask(rst_cls_idx)
        cover_idx = np.argwhere(cls_mask)

        # map coverage to target map with inat class indices
        target_map[inat_cls_idx, cover_idx[:, 0], cover_idx[:, 1]] = True

    # condense map to 1 layer with class indices
    target_map = np.argmax(target_map, axis=0)
    if debug:
        # check if targets are assigned correctly
        for inat_class in inat_classes:
            assert np.sum(ds.targets == inat_cls_idx_to_rst_cls_idx[inat_class_to_idx[inat_class]]) \
                   == target_map[inat_class_to_idx[inat_class]].sum()

    # get pixel positions of labeled area
    labeled_idx = np.argwhere(ds.get_labeled_area())

    # get target list and prediction list out of the labeled area
    targets = target_map[labeled_idx[:, 0], labeled_idx[:, 1]]
    preds = pred_map[labeled_idx[:, 0], labeled_idx[:, 1]]

    # balance classes by targets
    label_dist = dict(Counter(targets))
    label_dist = {inat_classes[k]: v for k, v in label_dist.items()}
    n = min(label_dist.values())

    # cut over-represented classes
    cls_labeled_idx = [np.argwhere(targets == inat_class_to_idx[inat_class])[:n] for inat_class in inat_classes]

    # combine the indices again
    eval_idx = np.concatenate(cls_labeled_idx, axis=0)

    # select samples from targets and predictions
    targets = targets[eval_idx]
    preds = preds[eval_idx]

    target_dist = dict(Counter(targets.reshape(-1)))
    target_dist = {inat_classes[k]: v for k, v in target_dist.items()}
    logging.info(f"Target class distribution: {target_dist}")

    pred_dist = dict(Counter(preds.reshape(-1)))
    pred_dist = {inat_classes[k]: v for k, v in pred_dist.items()}
    logging.info(f"Prediction class distribution: {pred_dist}")

    # convert to tensors
    targets = torch.IntTensor(targets)
    preds = torch.IntTensor(preds)

    # Calculate F1
    f1 = F1Score(num_classes=3)
    f1_score = float(f1(torch.IntTensor(preds), torch.IntTensor(targets)))

    # create confusion matrix
    cm = confusion_matrix(preds, targets, num_classes=3)
    df_cm = pd.DataFrame(cm.numpy(), index=range(3), columns=range(3))

    prediction_name = os.path.basename(pred_file)
    if result_dir is None:
        result_dir = os.path.dirname(pred_file)
    result_filename = f"{os.path.splitext(prediction_name)[0]}_pixel-confmat.png"

    # save figure
    plt.figure()
    sns.heatmap(df_cm, annot=True, cmap='Spectral', fmt='g').get_figure()
    plt.title(f"F1Score: {f1_score}")
    plt.savefig(os.path.join(result_dir, result_filename))

    return df_cm, pd.DataFrame({'predictions': preds.numpy().flatten(), 'targets': targets.numpy().flatten()}), f1_score


def optimize_image_resnet(cnn: InatClassifier, stages: list[int], norm_module: nn.Module, target_image: torch.Tensor,
                          loss_class: str, num_steps: int = 300, cuda: Optional[bool] = None):
    """
    Create an image that approximates the activation in the last layer of a given CNN model slice that is created by
    feeding a given target image to that model in terms of either the content loss or the style loss from the Neural
    Style Transfer paper by Gatys et al.
    WARNING: The used LBFGS optimizer doesn't work properly on GPU apparently. Sometimes the values of `x` become all
        NaN and the processing time per optimization step increases. Whenever that happens, just restart the process
        until it doesn't happen.
        Further information:
            - https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html
            - https://discuss.pytorch.org/t/l-bfgs-optimizer-doesnt-work-properly-on-cuda/136785
    :param cnn: The CNN ResNet CNN to be investigated.
    :param stages: The stages of the ResNet whose activation shall serve as target.
    :param norm_module: The parameterized normalization module.
    :param target_image: The image of which the activation shall be recreated by the image to be optimized.
    :param loss_class: Either 'ContentLoss' or 'StyleLoss'.
    :param num_steps: Number of steps taken for the optimization.
    :param cuda: Use cuda or not, default is the output of `torch.cuda.is_available()`.
    :return: The optimized image.
    """
    # prepare x and the target feature map
    x = torch.rand(target_image.shape).unsqueeze(0)

    # build model slice
    model = nn.Sequential()
    model.add_module('norm', norm_module)
    losses = []
    targets = []
    for stage_no, stage in enumerate(cnn.feature_extractor.children()):

        model.add_module(str(stage_no), stage)
        if isinstance(stage, nn.Sequential) and stage_no in stages:
            target = model(target_image.unsqueeze(0)).detach()
            targets.append(target)
            loss_module = eval(loss_class)(target)
            model.add_module(f"{loss_class.lower()}_{len(losses)}", loss_module)
            losses.append(loss_module)

            if stage_no == stages[-1]:
                break

    # now we trim off the layers after the last content and style losses
    # In my case this seems not to be necessary, but let's keep it for safety.
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    del i

    if cuda is None:
        cuda = torch.cuda.is_available()

    if cuda:
        model.double().cuda()
        x = x.double().cuda()
        for loss_module in losses:
            loss_module.double().cuda()

    # Activate gradient tracking for x and CNN slice
    x.requires_grad_(True)
    model.requires_grad_(True)

    # Create optimizer
    optimizer = optim.LBFGS([x])

    # Optimization loop
    pbar = tqdm(range(num_steps))
    pbar.set_description(f"Loss: {0}")
    for _ in pbar:

        def closure():
            with torch.no_grad():
                x.clamp_(0, 1)

            optimizer.zero_grad()
            model(x)
            loss = 0
            for l_module in losses:
                loss += l_module.loss
            loss.backward()

            pbar.set_description(f"Loss: {np.around(loss.detach().cpu().numpy(), 6)}")

            return sum([lm.loss for lm in losses])

        optimizer.step(closure)

    with torch.no_grad():
        x.clamp_(0, 1)

    # Move data and model back to CPU
    model.float().cpu()
    x = x.detach().float().cpu()
    for loss_module in losses:
        loss_module.float().cpu()

    return x


def visualize_integrated_gradients(x: torch.Tensor, pred_label_idx: torch.Tensor, model: nn.Module,
                                   transformed_img: torch.Tensor, out_dir: Union[Path, str],
                                   filename: Union[Path, str]):

    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(
        x,
        target=pred_label_idx,
        n_steps=200
    )

    cmap = LinearSegmentedColormap.from_list('custom blue', [(0, '#ffffff'), (.25, '#000000'), (1, '#000000')], N=256)
    ig_viz = viz.visualize_image_attr_multiple(
        np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        methods=['original_image', 'masked_image'],
        cmap=cmap,
        show_colorbar=True,
        signs=['all', 'positive'],
        outlier_perc=1,
        use_pyplot=False
    )[0]

    ig_viz.suptitle(f"{filename}, Integrated Gradients")
    ig_viz.savefig(os.path.join(out_dir, f"{filename}_integrated_gradients.png"))


def visualize_occlusion(x: torch.Tensor, pred_label_idx: torch.Tensor, model: nn.Module,
                        transformed_img: torch.Tensor, out_dir: Union[Path, str],
                        filename: Union[Path, str]):

    occlusion = Occlusion(model)
    attributions_occ = occlusion.attribute(
        x,
        strides=(3, 6, 6),
        target=pred_label_idx,
        sliding_window_shapes=(3, 12, 12),
        baselines=0,
        show_progress=True
    )

    occ_viz = viz.visualize_image_attr_multiple(
        np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        show_colorbar=True,
        outlier_perc=2,
        use_pyplot=False
    )[0]

    occ_viz.suptitle(f"{filename}, Occlusion")
    occ_viz.savefig(os.path.join(out_dir, f"{filename}_occlusion.png"))


def prepare_inputs(input_img: torch.Tensor, dm: InatDataModule):
    norm = dm.get_normalize_module()
    transformed_img = input_img.unsqueeze(0)
    x = norm(transformed_img)

    return x, transformed_img


def visualize_features(input_img: torch.Tensor, filename: Union[Path, str], model: nn.Module, dm: InatDataModule,
                       out_dir: Union[Path, str]):

    x, transformed_img = prepare_inputs(input_img, dm)

    out = model(x)
    prediction_socre, pred_label_idx = torch.topk(out, 1)

    visualize_integrated_gradients(x, pred_label_idx, model, transformed_img, out_dir, filename)
    visualize_occlusion(x, pred_label_idx, model, transformed_img, out_dir, filename)


def visualize_class_features(data_dir: Union[Path, str], model_path: Union[Path, str], preds_path: Union[Path, str],
                             out_dir: Optional[Union[Path, str]] = None, samples_per_class: int = 5,
                             model_class: str = 'InatSequentialClassifier'):

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(preds_path), "plots")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dm = InatDataModule(data_dir, normalize=False, batch_size=1, return_path=True)
    model = eval(model_class).load_from_checkpoint(model_path)
    _ = model.eval()
    pred_df = pd.read_csv(preds_path)

    for cls_name in dm.ds.classes:
        pred_df[f'{cls_name}_prob'] = pred_df[f'{cls_name}_prob'].astype(float)
        cls_idx = dm.ds.dataset.class_to_idx[cls_name]
        samples = pred_df[(pred_df.target == cls_idx) & (pred_df.prediction == cls_idx)].nlargest(
            samples_per_class, columns=[f'{cls_name}_prob'])
        sample_pids = samples.pid.values
        for pid in sample_pids:
            sample_img, _, _ = dm.ds.dataset.get_item_by_pid(pid)
            visualize_features(sample_img, pid, model, dm, out_dir)


def visualize_grad_cam(input_img: torch.Tensor, filename: Union[Path, str], model: nn.Module, target_layer,
                       dm: InatDataModule, out_dir: Union[Path, str]):

    x, transformed_img = prepare_inputs(input_img, dm)
    pred_scores = model(x).squeeze()

    gg_cam = GuidedGradCam(model, target_layer)
    attributions_cams = [gg_cam.attribute(x.requires_grad_(), t) for t in range(len(dm.ds.classes))]

    for t, attr_cam in enumerate(attributions_cams):
        cam_viz = viz.visualize_image_attr_multiple(
            np.transpose(attr_cam.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            ["original_image", "heat_map"],
            ["all", "positive"],
            show_colorbar=True,
            outlier_perc=2,
            use_pyplot=False
        )[0]

        cam_viz.suptitle(f"{filename}, GuidedGradCAM, {dm.ds.classes[t]}: {np.around(pred_scores[t].item(), 4)}")
        cam_viz.savefig(os.path.join(out_dir, f"{filename}_guided_grad_cam_{dm.ds.classes[t]}.png"))


def visualize_confusion_resnet(data_dir: Union[Path, str], model_path: Union[Path, str], preds_path: Union[Path, str],
                               out_dir: Optional[Union[Path, str]] = None, n_samples: Optional[int] = None):

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(preds_path), "plots")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dm = InatDataModule(data_dir, normalize=False, batch_size=1, return_path=True)
    model = InatSequentialClassifier.load_from_checkpoint(model_path)
    target_layer = [layer for layer in list(model.feature_extractor[-1][-1].children())
                    if isinstance(layer, nn.Conv2d)][-1]
    _ = model.eval()
    pred_df = pd.read_csv(preds_path)
    prob_columns = [col for col in pred_df.columns if col.endswith('_prob')]
    pred_df['prob_std'] = pred_df[prob_columns].std(1)

    if n_samples is None:
        pred_df['prob_std_zscore'] = np.abs(zscore(pred_df['prob_std']))
        prob_std_mean = pred_df.prob_std.mean()
        unconfident_samples = pred_df[(pred_df.prob_std_zscore > 3) & (pred_df.prob_std < prob_std_mean)]
    else:
        unconfident_samples = pred_df.loc[pred_df.prob_std.nsmallest(n_samples).index]

    pids = unconfident_samples.pid.values
    p_bar = tqdm(pids)
    p_bar.set_description("Visualizing confusion")
    for pid in p_bar:
        sample_img, *_ = dm.ds.dataset.get_item_by_pid(pid)
        visualize_grad_cam(sample_img, pid, model, target_layer, dm, out_dir)


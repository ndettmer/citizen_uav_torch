import pytest
from CitizenUAV.data import *
import shutil
import torch
import os


tmp_dir = os.path.join('.', 'tmp')
data_dir = os.path.join(tmp_dir, 'data')
n_samples = 10


def setup():
    download_data('Zostera marina', data_dir, n_samples)


def teardown():
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


@pytest.fixture(autouse=True)
def setup_and_teardown():
    print("\nDownloading images")
    setup()
    yield
    print("\nDeleting downloaded images")
    teardown()


@pytest.mark.parametrize('ds_idx', [np.random.randint(n_samples)])
def test_dataset(ds_idx):
    metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    ds = InatDataset(data_dir, metadata)
    assert len(ds) == n_samples
    img, label = ds[np.random.randint(n_samples)]
    assert isinstance(img, torch.Tensor)
    assert isinstance(label, np.int64)

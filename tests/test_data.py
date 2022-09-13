import pytest
import shutil
from CitizenUAV.data import *
from collections import Counter


tmp_dir = os.path.join('.', 'tmp')
data_dir = os.path.join(tmp_dir, 'data')
n_samples = 5
species = ['Zostera marina', 'Ruppia maritima']


def setup():
    for spec in species:
        download_data(spec, data_dir, n_samples)


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


def test_datamodule():
    batch_size = 4

    # test without species
    dm = InatDataModule(data_dir)
    # check balance
    samples_per_class = list(dict(Counter(dm.ds.targets)).values())
    for n in samples_per_class:
        # tolerance of +-1
        assert samples_per_class[0] - 1 <= n <= samples_per_class[0] + 1

    # test with species
    img_size = 128
    dm = InatDataModule(data_dir, species, batch_size=4, img_size=img_size)
    dm.setup()
    assert len(dm.train_ds) <= len(dm.metadata) * dm.split[0]
    assert len(dm.val_ds) <= len(dm.metadata) * dm.split[1]
    assert len(dm.test_ds) < len(dm.metadata)

    # test data loader
    dl = dm.train_dataloader()
    x, y = next(iter(dl))
    assert x.shape[0] == y.shape[0]
    assert x.shape[0] == batch_size
    assert x.shape[2] == x.shape[3]
    assert x.shape[2] == img_size

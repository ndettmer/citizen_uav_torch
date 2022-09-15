import pytest
import shutil
from CitizenUAV.data import *
from CitizenUAV.models import *
from collections import Counter
import re
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch


tmp_dir = os.path.join('.', 'tmp')
data_dir = os.path.join(tmp_dir, 'data')
n_samples = [5, 8]
species = ['Zostera marina', 'Ruppia maritima']


def setup():
    for n, spec in zip(n_samples, species):
        download_data(spec, data_dir, n)


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


def test_augmentation():
    target_n = 10
    offline_augmentation(data_dir, target_n)
    ds = ImageFolder(data_dir)
    samples_per_class = list(dict(Counter(ds.targets)).values())
    for n in samples_per_class:
        # tolerance of +-1
        assert samples_per_class[0] - 1 <= n <= samples_per_class[0] + 1
    assert len(ds) == len(ds.classes) * target_n

    # Assert that every augmented file is in the same directory as its original.
    copy_regex = re.compile(r'[0-9]*_[0-9]*\.png')
    for root, cls_dir, filenames in os.walk(data_dir):
        for filename in filenames:
            if copy_regex.match(filename):
                orig_filename = f"{filename.split('_')[0]}.png"
                assert os.path.exists(os.path.join(root, orig_filename))


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


def test_model():
    batch_size = 1
    img_size = 64
    n_classes = 2
    model = InatClassifier(n_classes, 'resnet18')
    x = torch.randn(1, 3, img_size, img_size)
    model.eval()
    y_hat = model(x)
    assert y_hat.shape[0] == batch_size
    assert y_hat.shape[1] == n_classes
    assert (y_hat.flatten(1) <= 1).all()
    assert (y_hat.flatten(1) >= 0).all()


def test_training():
    dm = InatDataModule(data_dir, species, img_size=64)
    model = InatClassifier(len(species), 'resnet18')
    logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(data_dir, 'pl_logs'))
    trainer = pl.Trainer(max_epochs=1, max_steps=3, logger=logger)
    trainer.fit(model, dm)

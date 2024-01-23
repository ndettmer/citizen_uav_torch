import pytorch_lightning as pl
import torch.optim
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torchvision.models import resnet18, resnet50, wide_resnet50_2
from resnest.torch import resnest50
# TODO: Understand how RevCol can be used
#from RevCol import *
from MogaNet import *
from MogaNet.moganet import MogaNet
from torch import nn
from torch.nn import functional as F
from torchmetrics import F1Score, Accuracy
# precision_recall seems to have been moved or removed in newer torchmetrics versions
from torchmetrics.functional import precision_recall, confusion_matrix
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from typing import Optional, Union, Any
from pathlib import Path
from abc import ABC, abstractmethod


class InatClassifier(pl.LightningModule, ABC):

    @staticmethod
    @abstractmethod
    def add_model_specific_args(parent_parser):
        pass

    def __init__(self, n_classes: int, log_train_preds: bool = False, *args: Any, **kwargs: Any):
        super().__init__()
        self.n_classes = n_classes
        self.log_train_preds = log_train_preds

        self.loss_function = nn.CrossEntropyLoss()
        self.f1 = F1Score(num_classes=self.n_classes, average='macro')
        self.acc = Accuracy(num_classes=self.n_classes, average='macro')

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        loss = self.loss_function(y_hat, y)
        self.log_dict({"train_cce": loss})

        return {'train_preds': preds, 'train_targets': y, 'loss': loss, 'train_y_hat': y_hat}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log_dict({"val_cce": loss})

        preds = torch.argmax(y_hat, dim=1)

        return {'val_preds': preds, 'val_targets': y, 'loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log_dict({"test_cce": loss})

        preds = torch.argmax(y_hat, dim=1)

        return {'test_preds': preds, 'test_targets': y, 'loss': loss}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        y_hat = torch.concat([o['train_y_hat'] for o in outputs])
        preds = torch.concat([o['train_preds'] for o in outputs])
        targets = torch.concat([o['train_targets'] for o in outputs])
        train_f1 = self.f1(preds, targets)
        train_prec, train_rec = precision_recall(preds, targets, num_classes=self.n_classes)
        train_acc = self.acc(preds, targets)
        log_content = {
            'train_f1': train_f1,
            'train_prec': train_prec,
            'train_rec': train_rec,
            'train_acc': train_acc
        }
        if self.log_train_preds:
            log_content['train_y_hat'] = y_hat
            log_content['train_targets'] = targets
        self.log_dict(log_content)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        preds = torch.concat([o['val_preds'] for o in outputs])
        targets = torch.concat([o['val_targets'] for o in outputs])
        val_f1 = self.f1(preds, targets)
        val_prec, val_rec = precision_recall(preds, targets, num_classes=self.n_classes)
        val_acc = self.acc(preds, targets)
        self.log_dict({
            'val_f1': val_f1,
            'val_prec': val_prec,
            'val_rec': val_rec,
            'val_acc': val_acc
        })

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        preds = torch.concat([o['test_preds'] for o in outputs])
        targets = torch.concat([o['test_targets'] for o in outputs])
        test_f1 = self.f1(preds, targets)
        test_prec, test_rec = precision_recall(preds, targets, num_classes=self.n_classes, average='macro')
        test_acc = self.acc(preds, targets)
        conf_mat = confusion_matrix(preds, targets, num_classes=self.n_classes)
        self.log_dict({
            'test_f1': test_f1,
            'test_prec': test_prec,
            'test_rec': test_rec,
            'test_acc': test_acc
        })

        df_cm = pd.DataFrame(conf_mat.cpu().numpy(), index=range(self.n_classes), columns=range(self.n_classes))
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral', fmt='g').get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("Confusion matrix/Test", fig_, self.current_epoch)


class InatSequentialClassifier(InatClassifier):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("InatSequentialClassifier")
        parser.add_argument("--n_classes", type=int)
        parser.add_argument("--backbone_model", type=str, default='resnet18')
        parser.add_argument("--weights", type=str, default=None,
                            help="String key of the weights to be loaded from torchvision.")
        parser.add_argument("--checkpoint_path", type=str, default=None,
                            help="Path to a pre-trained backbone model checkpoint.")
        parser.add_argument("--hub_link", type=str, default=None,
                            help="Relative path to torch hub weights.")
        parser.add_argument("--lr", type=float, required=False, default=.0001)
        parser.add_argument("--weight_decay", type=float, required=False, default=.001)
        parser.add_argument("--log_train_preds", type=bool, required=False, default=False)
        parser.add_argument("--native_classifier", type=bool, required=False, default=False)
        return parent_parser

    def __init__(self, n_classes, backbone_model, lr, weight_decay, log_train_preds: bool = True,
                 weights: Optional[str] = None, checkpoint_path: Optional[Union[str, Path]] = None,
                 hub_link: Optional[str] = None, **kwargs):

        if sum([weights is not None, checkpoint_path is not None, hub_link is not None]) > 1:
            raise ValueError(f"Only one argument of weights and weight_path can be given.")

        super().__init__(n_classes=n_classes, log_train_preds=log_train_preds)
        self.save_hyperparameters()

        # backbone
        if checkpoint_path is not None:
            backbone = eval(backbone_model).load_from_checkpoint(checkpoint_path)
        elif hub_link is not None:
            backbone = torch.hub.load(hub_link, backbone_model, pretrained=True)
        elif weights is not None:
            backbone = eval(backbone_model)(weights=weights)
        else:
            # default is weights=None
            backbone = eval(backbone_model)(num_classes=n_classes)

        if 'moganet' in backbone_model:
            n_backbone_features = backbone.head.in_features
        else:
            # Default is ResNet architecture
            n_backbone_features = backbone.fc.in_features

        self.native_classifier = kwargs.get('native_classifier', True) and not bool(weights)
        if self.native_classifier:
            self.feature_extractor = backbone
            self.classifier = nn.Softmax(dim=1)
        else:
            fe_layers = list(backbone.children())[:-2]
            self.feature_extractor = nn.Sequential(*fe_layers)
            self.classifier = nn.Sequential(
                nn.Linear(n_backbone_features, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, n_classes),
                nn.Softmax(dim=1)
            )

        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        # 1. feature extraction
        x = self.feature_extractor(x)
        if not self.native_classifier:
            # 2. global max pooling
            x = F.max_pool2d(x, kernel_size=x.size()[2:]).flatten(1)
        # 3. classification
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        # weight decay > 0 is the L2 regularization
        return torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class InatMogaNetClassifier(InatClassifier):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("InatMogaNetClassifier")
        parser.add_argument("--model_size", type=str, required=True, choices=['small', 'tiny', 'base', 'large'])

        # optimizer configuration taken from
        # https://github.com/Westlake-AI/openmixup/blob/main/configs/classification/imagenet/moganet/moga_small_ema_sz224_8xb128_ep300.py
        # fit for MogaNet small
        parser.add_argument("--lr", type=float, default=1e-3, required=False)
        parser.add_argument("--weight_decay", type=float, default=.05, required=False)
        parser.add_argument("--admw_eps", type=float, default=1e-8, required=False)
        parser.add_argument("--admw_beta1", type=float, default=.9, required=False)
        parser.add_argument("--admw_beta2", type=float, default=.999, required=False)

        return parent_parser

    def __init__(self, n_classes, model_size, lr=1e-3, weight_decay=.05, adamw_eps=1e-8, adamw_beta1=.9,
                 adamw_beta2=.999, **kwargs):
        super().__init__(n_classes=n_classes)
        self.save_hyperparameters()
        self.moganet = eval(f"moganet_{model_size}")(num_classes=self.n_classes)
        self.lr = lr
        self.weight_decay = weight_decay
        self.adamw_eps = adamw_eps
        self.adamw_beta1 = adamw_beta1
        self.adamw_beta2 = adamw_beta2
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.moganet(x)
        x = self.softmax(x)
        return x

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.moganet.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=self.adamw_eps,
            betas=(self.adamw_beta1, self.adamw_beta2)
        )


class InatRevColClassifier(pl.LightningModule):
    # TODO

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        pass


class InatRegressor(pl.LightningModule):
    # Only Distance is needed!!
    # Filtering by angle had no positive effects!

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("InatRegressor")
        parser.add_argument("--backbone_model", type=str, default='resnet18')
        return parent_parser

    def __init__(self, backbone_model, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # backbone
        # default is weights=None
        backbone = eval(f"{backbone_model}()")
        n_backbone_features = backbone.fc.in_features
        fe_layers = list(backbone.children())[:-2]
        self.feature_extractor = nn.Sequential(*fe_layers)

        self.regressor = nn.Sequential(
            nn.Dropout(p=.5),
            nn.Linear(n_backbone_features, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. feature extraction
        features = self.feature_extractor(x)
        # 2. global max pooling
        x = F.max_pool2d(features, kernel_size=features.size()[2:]).flatten(1)
        # 3. regression
        x = self.regressor(x)
        return x

    def configure_optimizers(self):
        # weight decay > 0 is the L2 regularization
        return torch.optim.Adam(self.parameters(), lr=.0001, weight_decay=.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().unsqueeze(-1)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log_dict({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().unsqueeze(-1)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log_dict({"val_loss": loss})
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().unsqueeze(-1)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log_dict({"test_loss": loss})
        return loss


class RandomClassifier(pl.LightningModule):
    """
    Dummy model for testing predictions scripts.
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser.add_argument("--n_classes", type=int, required=True)
        return parent_parser

    def __init__(self, **kwargs):
        super().__init__()
        self.n_classes = kwargs.pop('n_classes')

    def forward(self, x):
        batch_size = x.shape[0]
        return torch.rand((batch_size, self.n_classes))


def count_parameters(model: nn.Module):
    """
    source: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

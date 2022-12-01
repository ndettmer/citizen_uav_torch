import pytorch_lightning as pl
import torch.optim
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torchvision.models import resnet18, resnet50, wide_resnet50_2
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, fasterrcnn_resnet50_fpn
from torch import nn
from torch.nn import functional as F
from torchmetrics import F1Score, Accuracy
from torchmetrics.functional import precision_recall, confusion_matrix
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


class InatClassifier(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("InatClassifier")
        parser.add_argument("--n_classes", type=int)
        parser.add_argument("--backbone_model", type=str, default='resnet18')
        parser.add_argument("--lr", type=float, required=False, default=.0001)
        parser.add_argument("--weight_decay", type=float, required=False, default=.001)
        return parent_parser

    def __init__(self, n_classes, backbone_model, lr, weight_decay, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # backbone
        # default is weights=None
        backbone = eval(f"{backbone_model}()")
        n_backbone_features = backbone.fc.in_features
        fe_layers = list(backbone.children())[:-2]
        self.feature_extractor = nn.Sequential(*fe_layers)

        # TODO try out less complex classifier
        self.classifier = nn.Sequential(
            nn.Linear(n_backbone_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
            nn.Softmax(dim=1)
        )

        self.n_classes = n_classes

        self.loss_function = nn.CrossEntropyLoss()
        self.f1 = F1Score(num_classes=n_classes, average='macro')
        self.acc = Accuracy(num_classes=n_classes, average='macro')

        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        # 1. feature extraction
        features = self.feature_extractor(x)
        # 2. global max pooling
        x = F.max_pool2d(features, kernel_size=features.size()[2:]).flatten(1)
        # 3. classification
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        # weight decay > 0 is the L2 regularization
        return torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        loss = self.loss_function(y_hat, y)
        self.log_dict({"train_cce": loss})

        return {'train_preds': preds, 'train_targets': y,  'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log_dict({"val_cce": loss})

        preds = torch.argmax(y_hat, dim=1)

        return {'val_preds': preds, 'val_targets': y,  'loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log_dict({"test_cce": loss})

        preds = torch.argmax(y_hat, dim=1)

        return {'test_preds': preds, 'test_targets': y,  'loss': loss}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        preds = torch.concat([o['train_preds'] for o in outputs])
        targets = torch.concat([o['train_targets'] for o in outputs])
        train_f1 = self.f1(preds, targets)
        train_prec, train_rec = precision_recall(preds, targets, num_classes=self.n_classes)
        train_acc = self.acc(preds, targets)
        self.log_dict({
            'train_f1': train_f1,
            'train_prec': train_prec,
            'train_rec': train_rec,
            'train_acc': train_acc
        })

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


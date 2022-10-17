import pytorch_lightning as pl
import torch.optim
from torchvision.models import resnet18, resnet50, wide_resnet50_2
from torch import nn
from torch.nn import functional as F


class InatClassifier(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("InatClassifier")
        parser.add_argument("--n_classes", type=int)
        parser.add_argument("--backbone_model", type=str, default='resnet18')
        return parent_parser

    def __init__(self, n_classes, backbone_model, **kwargs):
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
            nn.Softmax()
        )

        self.loss_function = nn.CrossEntropyLoss()

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
        return torch.optim.RMSprop(self.parameters(), lr=.0001, weight_decay=.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log_dict({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log_dict({"val_loss": loss})
        return loss


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


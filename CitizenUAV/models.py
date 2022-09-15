import pytorch_lightning as pl
import torch.optim
from torchvision.models import resnet18, resnet50, wide_resnet50_2
from torch import nn
import torch.nn.functional as F


class InatClassifier(pl.LightningModule):
    def __init__(self, n_classes, backbone_model):
        super().__init__()

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
        loss = F.cross_entropy(y_hat, y)
        return loss

import torch.nn as nn
from models.camdt import CAMDT
from models.amdt import AMDT
from models.bbt import BBT

class CAMDT_BBT(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = CAMDT()
        self.amdt = AMDT()
        self.bbt = BBT()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.amdt(x)
        x = self.bbt(x)

        x = self.pool(x).flatten(1)
        return self.head(x)
import torch
import torch.nn as nn
from config import *

class Discriminator(nn.Module):
    def __init__(self, ndf=NDF, num_classes=NUM_CLASSES, dropout_p=DROPOUT):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1),       
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_p),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1), 
            nn.GroupNorm(1, ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_p),

            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1),
            nn.GroupNorm(1, ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_p),
        )

        self.adv_layer = nn.Conv2d(ndf * 4, 1, 7, 1, 0)        
        self.aux_layer = nn.Conv2d(ndf * 4, num_classes, 7, 1, 0)
        self.width_layer = nn.Conv2d(ndf * 4, 1, 7, 1, 0)        

        self.initialize_weights() 
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        feat = self.conv(x)             # (B, ndf*4, 7, 7)

        validity = self.adv_layer(feat) # (B, 1, 1, 1)
        validity = validity.view(-1)    # (B,)

        label = self.aux_layer(feat)    # (B, NUM_CLASSES, 1, 1)
        label = label.view(-1, NUM_CLASSES)  # (B, NUM_CLASSES)

        width = self.width_layer(feat)  # (B, 1, 1, 1)
        width = width.view(-1)          # (B,)

        return validity, label, width

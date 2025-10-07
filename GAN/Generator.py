import torch
import torch.nn as nn
from config import *

class Generator(nn.Module):
    def __init__(self, nz=NZ, ngf=NGF, num_classes=NUM_CLASSES, dropout_p=DROPOUT):
        super().__init__()
        self.embed = nn.Embedding(num_classes, nz)
        self.fc_width = nn.Linear(1, nz)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(3 * nz, ngf * 4, 7, 1, 0), 
            nn.GroupNorm(1, ngf * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_p),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),  
            nn.GroupNorm(1, ngf * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_p),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1), 
            nn.GroupNorm(1, ngf),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_p),

            nn.Conv2d(ngf, 1, 3, 1, 1), 
            nn.Tanh()
        )
        self.initialize_weights() 
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, z, labels, width):
        # z: (B, nz,)
        # labels: (B,)
        # width: (B,)

        y = self.embed(labels)                  # (B, nz)
        w = self.fc_width(width.view(-1, 1))    # (B, nz)

        x = torch.cat([z, y, w], 1)             # (B, 3*nz)
        x = x.view(x.size(0), x.size(1), 1, 1)  # (B, 3*nz, 1, 1)

        return self.main(x)                     # (B, 1, 28, 28)

from imbalanced import ImbalancedDatasetSampler
from collections import Counter
import os
from tqdm import tqdm
import numpy as np 
import pandas
import torch
import pytorch_lightning as pl
from torchvision import models, transforms
from config_file import config
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from torch.utils.data.dataset import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from sklearn import metrics
import pandas as pd
from PIL import Image
import cv2
import timm

class MelanomaDetector(pl.LightningModule):
    def __init__(self, train_dl=None, val_dl=None):
        super(MelanomaDetector, self).__init__()                
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.1))
        self.learning_rate = 0.001     
        self.train_dl, self.val_dl = train_dl, val_dl
        self.model = timm.create_model('resnet50', pretrained=True, num_classes=1, drop_rate=0.5)
        
    def forward(self, x):
        x = self.model(x)        
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        # print (y)
        y_hat = self.forward(x).view(y.size())
        loss = self.criterion(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x).view(y.size())
        return {'val_loss': self.criterion(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x).view(y.size())
        return {'test_loss': self.criterion(y_hat, y)}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.val_dl
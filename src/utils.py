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

augmentations = {
    'train':A.Compose([
            A.HorizontalFlip(p = 0.5),
            A.OneOf([
            A.RandomContrast(),
            A.RandomGamma(),
            A.RandomBrightness()],
            p = 0.5),            
            A.Normalize(mean=config['mean'], std=config['std']),            
            A.pytorch.ToTensor()],
            p = 1),
    'valid':A.Compose([            
            A.Normalize(mean=config['mean'], std=config['std']),            
            A.pytorch.ToTensor()])
}
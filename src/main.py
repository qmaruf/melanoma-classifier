import sys
sys.path.insert(0, '../src/')
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
from model import MelanomaDetector
from utils import augmentations

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-sample','--sample', help='use a smaller part of data', action='store_true')
parser.add_argument('-phase','--phase', help='train/test the trained model', choices=['train', 'test'], required=True)
parser.add_argument('-model_path','--model_path', help='load a trained model')
args = vars(parser.parse_args())

to_pil = transforms.ToPILImage()

checkpoint_callback = ModelCheckpoint(
    filepath=os.getcwd(),
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix='melanoma'
)

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.001,
   patience=5,
   verbose=True,
   mode='min',
)

class MelanomaDataset(Dataset):
    def __init__(self, images, labels, transforms):
        self.images = images
        self.labels = labels
        self.transforms = transforms
        
    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        label = self.labels[index]*1.0
        return image, label

    def __len__(self):
        return len(self.images)
    

def get_data_loader(fold=0):
    
    df = pd.read_csv(os.path.join(config['data_dir'], 'train_folds.csv'))
    df['image_name'] = df['image_name'].apply(lambda x: '../input/224x224/%s.png'%x)

    train_idx = df[df.kfold != fold].index.values
    valid_idx = df[df.kfold == fold].index.values

    train_imgs, train_lbls = df['image_name'][train_idx].to_list(), df['target'][train_idx].to_list()
    valid_imgs, valid_lbls = df['image_name'][valid_idx].to_list(), df['target'][valid_idx].to_list()

    print (len(train_imgs), len(valid_imgs), Counter(train_lbls), Counter(valid_lbls))
    # exit()
    
    if args['sample']:
        train_imgs, train_lbls = train_imgs[:100], train_lbls[:100]
        valid_imgs, valid_lbls = valid_imgs[:100], valid_lbls[:100]
    
    
    train_ds = MelanomaDataset(train_imgs, train_lbls, augmentations['train'])
    train_dl = DataLoader(train_ds, batch_size=config['train_bs'], shuffle=False, sampler=ImbalancedDatasetSampler(train_ds))
    # train_dl = DataLoader(train_ds, batch_size=config['train_bs'], shuffle=True)

    valid_ds = MelanomaDataset(valid_imgs, valid_lbls, augmentations['valid'])
    valid_dl = DataLoader(valid_ds, batch_size=config['val_bs'], shuffle=False)
    
    return train_dl, valid_dl

def train(fold):      
    train_dl, valid_dl = get_data_loader(fold)   
    
    model = MelanomaDetector(train_dl, valid_dl)
    
    trainer = pl.Trainer(max_epochs=config['epochs'],
                        gpus=1, 
                        check_val_every_n_epoch=5,
                        auto_lr_find=True,
                        callbacks=[early_stop_callback, checkpoint_callback])    
    trainer.fit(model)


def test():
    train_dl, valid_dl = get_data_loader(config['fold'])
    print (len(train_dl.dataset))
    # exit()
    model = MelanomaDetector.load_from_checkpoint(checkpoint_path=args['model_path'])
    model.eval()
    
    y_pred, y_true = [], []
    for batch in tqdm(valid_dl):
        imgs, lbls = batch
        # img = to_pil(imgs[0].data.cpu())
        # plt.figure()
        # plt.imshow(img)
        # plt.savefig('img.jpg')
        # print (x.size())
        # exit()
        preds = nn.Sigmoid()(model(imgs))
        # print (lbls)
        # print(np.around(preds.data.cpu().numpy(), decimals=3))

        # print('----')
        y_pred.append(preds.data.cpu().numpy())
        y_true.append(lbls.data.cpu().numpy())
            # print (lb)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
    plt.figure()
    plt.plot(recall, precision)
    plt.grid()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.savefig('./precision_recall_curve.jpg')

    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.grid()
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.savefig('./roc_curve.jpg')

    auroc = metrics.roc_auc_score(y_true, y_pred)
    print ('roc_auc', auroc)

if __name__ == '__main__':
    if args['phase'] == 'train':
        train(config['fold'])
    else:
        if not Path(args['model_path']).is_file():
            raise Exception('checkpoint %s does not exist'%args['model_path'])
        test()
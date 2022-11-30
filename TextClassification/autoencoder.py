from doctest import Example
from genericpath import samefile
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import pandas as pd
from sklearn import preprocessing 
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import seaborn as sns
from matplotlib import pyplot as plt
import umap
import numpy as np
from pytorch_lightning.loggers.neptune import NeptuneLogger
from sklearn.model_selection import train_test_split
from pytorch_lightning import loggers as pl_loggers
from tqdm import tqdm
from argparse import ArgumentParser
from torch.optim.lr_scheduler import ReduceLROnPlateau

class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index, :][:, np.newaxis]
        # print(x.shape)
        if self.transform:
            x = self.transform(x)
        return x
    
    def __len__(self):
        return self.data.shape[0]

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        # Include extra logging here
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        # Include extra logging here
        self.log('valid_loss', loss, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        # Include extra logging here
        self.log('test_loss', loss, on_epoch=True)
        return loss

    # ---------------------
    # training setup
    # ---------------------
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = optim.AdamW(self.parameters(), lr=0.001)


        scheduler = ReduceLROnPlateau(optimizer, mode='min')
        metric_to_track = 'valid_loss'
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': metric_to_track,
                'strict': False
            },
        }

    def configure_callbacks(self):
        early_stop = pl.callbacks.EarlyStopping(monitor="valid_loss", mode="min", patience=10)
        checkpoint = pl.callbacks.ModelCheckpoint(monitor="valid_loss", save_top_k=5,
                                                  dirpath='ckpt',
                                                  filename='Autoencoder-{epoch:02d}-{valid_loss:.5f}')
        return [early_stop, checkpoint]


def read_embed(datadir):

    training_features_npy = datadir + '/training_features_umap_400.npy'
    testing_features_npy = datadir + '/testing_features_umap_400.npy'
    val_features_npy = datadir + '/val_features_umap_400.npy'

    if os.path.exists(training_features_npy):
        training_embedding = np.load(training_features_npy)
        testing_embedding = np.load(testing_features_npy)
        val_embedding = np.load(val_features_npy)
    else:
        raise Exception("Please read split_data.py first to generate data!")

    return training_embedding, val_embedding, testing_embedding


def cli_main():

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_dataloader_workers', type=int, default=16)
    parser.add_argument('--datadir', type=str, default='../data')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    

    encoder = nn.Sequential(nn.Linear(400, 64), nn.ReLU(), nn.Linear(64, 10))
    decoder = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 400))
    ae = LitAutoEncoder(encoder, decoder)

    # df = read_data()

    # training_set, test_set, training_labels, test_labels = train_test_split(df["filtered_content"], df["category"], test_size=0.33)

    # training_set, val_set, training_labels, val_labels = train_test_split(training_set, training_labels, test_size=0.2)

    train_embed, val_embed, test_embed = read_embed(args.datadir)    

    ##parameters for loading models for each target
    train_params = {'batch_size': 16,

                    'shuffle': True,

                    'num_workers': 16}

    val_params = {'batch_size': 16,

                    'num_workers': 16}
    
    test_params = {'batch_size': 16,

                    'shuffle': True,

                    'num_workers': 16}

    train_loader = DataLoader(MyDataset(train_embed, ToTensor()), **train_params)

    val_loader = DataLoader(MyDataset(val_embed, ToTensor()), **val_params)

    test_loader = DataLoader(MyDataset(test_embed, ToTensor()), **test_params)
    
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)

    trainer.fit(model=ae, train_dataloaders=train_loader)

    # ------------
    # training
    # ------------
    
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")

    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger)

    trainer.fit(ae, train_loader, val_loader)

    result = trainer.test(model=ae, test_dataloaders=test_loader)
    
    print(result)


if __name__ == '__main__':
    cli_main()

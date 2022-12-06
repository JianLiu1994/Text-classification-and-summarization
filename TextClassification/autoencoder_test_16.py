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
import numpy as np
from pytorch_lightning.loggers.neptune import NeptuneLogger
from sklearn.model_selection import train_test_split
from pytorch_lightning import loggers as pl_loggers
from tqdm import tqdm
from argparse import ArgumentParser
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.cluster import KMeans
from sklearn import metrics, neighbors
import pickle
from sklearn.metrics import classification_report

class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        #print(self.data[index, :].shape)
        x = self.data[index, :][:, np.newaxis]
        # print(x.shape)
        if self.transform:
            x = self.transform(x)
        return x
    
    def __len__(self):
        return self.data.shape[0]

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder, outdir, test_labels):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.outdir = outdir
        self.test_labels = test_labels

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
        return {'test_loss' : loss, 'latents': z.cpu().detach().numpy().astype(np.float32)}

    def test_epoch_end(self, outputs):
        test_embeddings = []
        for output in outputs:
            embedding = output['latents']
            test_embeddings += [embedding]
        test_embeddings = np.squeeze(np.array(test_embeddings),1)
        print(test_embeddings.shape)

        km_model = KMeans(n_clusters=4, n_init=50, max_iter=1000)
        km_model.fit(test_embeddings)
    
        print(classification_report(self.test_labels, km_model.labels_, target_names=['business', 'entertainment', 'health', 'science and technology']))
        

    # ---------------------
    # training setup
    # ---------------------
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = optim.AdamW(self.parameters(), lr=0.00001)
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
                                                  dirpath=self.outdir + '/ckpt',
                                                  filename='Autoencoder-{epoch:02d}-{valid_loss:.5f}')
        return [early_stop, checkpoint]


def read_embed(datadir):

    training_features_pkl = datadir + '/training_features_umap_3000.pkl'
    testing_features_pkl = datadir + '/test_features_umap_3000.pkl'
    val_features_pkl = datadir + '/val_features_umap_3000.pkl'

    if not os.path.exists(training_features_pkl):
        raise Exception("Please read split_data.py first to generate data!")

    training_embedding = pickle.load(open(training_features_pkl, "rb" ) )
    val_embedding = pickle.load(open(val_features_pkl, "rb" ) )
    testing_embedding = pickle.load(open(testing_features_pkl, "rb" ) )

    return training_embedding, val_embedding, testing_embedding


def cli_main():

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_dataloader_workers', type=int, default=16)
    parser.add_argument('--datadir', type=str, default='../data')
    parser.add_argument('--outdir', type=str, default='.')
    parser.add_argument('--ckpt_name', type=str, default="")
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    train_embed, val_embed, test_embed = read_embed(args.datadir)    
    test_labels = np.load(args.datadir + '/test_labels.npy')

    ##parameters for loading models for each target
    train_params = {'batch_size': 64,

                    'shuffle': True,

                    'num_workers': 16}

    val_params = {'batch_size': 64,

                    'num_workers': 16}
    
    test_params = {'batch_size': 1,

                    'num_workers': 16}

    print(train_embed.shape)
    print(val_embed.shape)
    print(test_embed.shape)

    train_loader = DataLoader(MyDataset(train_embed, ToTensor()), **train_params)

    val_loader = DataLoader(MyDataset(val_embed, ToTensor()), **val_params)

    test_loader = DataLoader(MyDataset(test_embed, ToTensor()), **test_params)

    encoder = nn.Sequential(nn.Linear(3000, 2048), nn.ReLU(), nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 16))
    decoder = nn.Sequential(nn.Linear(16, 1024), nn.ReLU(), nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 3000))
    ae = LitAutoEncoder(encoder, decoder, args.outdir, test_labels)


    # ------------
    # training
    # ------------

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.outdir + '/logs/')

    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, resume_from_checkpoint=args.ckpt_name, max_epochs=399)

    # trainer.resume_from_checkpoint = args.ckpt_name if os.path.exists(args.ckpt_name) else None

    trainer.fit(ae, train_loader, val_loader)

    result = trainer.test(model=ae, test_dataloaders=test_loader)
    
    print(result)


if __name__ == '__main__':
    cli_main()

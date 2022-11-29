from genericpath import samefile
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
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

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.001)
        return optimizer

def read_data():
    df = pd.read_pickle('../Data_collection/dataset_filtered.pickle')
    label_encoder = preprocessing.LabelEncoder()
    df['category']= label_encoder.fit_transform(df['category']) 

    df0 = df[df.category==0]
    df1 = df[df.category==1]
    df2 = df[df.category==2]
    df3 = df[df.category==3]

    samples = df.category.value_counts().tolist()
    sample_count = np.min(np.array(samples))
    df0 = resample(df0, replace=True, n_samples=sample_count, random_state=123)
    df1 = resample(df1, replace=True, n_samples=sample_count, random_state=123)
    df2 = resample(df2, replace=True, n_samples=sample_count, random_state=123)
    df3 = resample(df3, replace=True, n_samples=sample_count, random_state=123)
    df_sampled = pd.concat([df0,df1,df2,df3])
    return df_sampled

def embed(training_set, val_set, test_set):

    training_features_npy = 'training_features_400.npy'
    testing_features_npy = 'testing_features_400.npy'
    val_features_npy = 'val_features_400.npy'

    if os.path.exists(training_features_npy):
        training_embedding = np.load(training_features_npy)
        testing_embedding = np.load(testing_features_npy)
        val_embedding = np.load(val_features_npy)
    else:
        tfidvectorizer = TfidfVectorizer(min_df=2, 
                                        ngram_range=(2,2),
                                        smooth_idf=True,
                                        use_idf=True)

        tfid_train_features = tfidvectorizer.fit_transform(training_set)
        tfid_val_features = tfidvectorizer.transform(val_set)
        tfid_test_features = tfidvectorizer.transform(test_set)

        reducer = umap.UMAP(random_state=42, n_components=400)
        training_embedding = reducer.fit_transform(tfid_train_features)
        val_embedding = reducer.transform(tfid_val_features)
        testing_embedding = reducer.transform(tfid_test_features)

        np.save(training_features_npy, training_embedding)
        np.save(val_features_npy, val_embedding)
        np.save(testing_features_npy, testing_embedding)

    return training_embedding, val_embedding, testing_embedding


def cli_main():

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_dataloader_workers', type=int, default=16)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    encoder = nn.Sequential(nn.Linear(400, 64), nn.ReLU(), nn.Linear(64, 10))
    decoder = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 400))
    ae = LitAutoEncoder(encoder, decoder)

    df = read_data()

    training_set, test_set, training_labels, test_labels = train_test_split(df["filtered_content"], df["category"], test_size=0.33)

    training_set, val_set, training_labels, val_labels = train_test_split(training_set, training_labels, test_size=0.2)

    train_embed, val_embed, test_embed = embed(training_set, val_set, test_set)    

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
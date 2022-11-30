import os
import pandas as pd
from sklearn import preprocessing 
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import seaborn as sns
from matplotlib import pyplot as plt
import umap
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import os, sys, argparse, time

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
    df0 = resample(df0, replace=True, n_samples=sample_count)
    df1 = resample(df1, replace=True, n_samples=sample_count)
    df2 = resample(df2, replace=True, n_samples=sample_count)
    df3 = resample(df3, replace=True, n_samples=sample_count)
    df_sampled = pd.concat([df0,df1,df2,df3])
    return df_sampled


def embed(training_set, val_set, test_set, outdir):

    tfidvectorizer = TfidfVectorizer(min_df=2, 
                                        ngram_range=(2,2),
                                        smooth_idf=True,
                                        use_idf=True)

    tfid_train_features = tfidvectorizer.fit_transform(training_set)
    tfid_val_features = tfidvectorizer.transform(val_set)
    tfid_test_features = tfidvectorizer.transform(test_set)

    training_features_npy = 'training_features.npy'
    val_features_npy = 'val_features.npy'
    testing_features_npy = 'test_features.npy'

    np.save(outdir + '/' + training_features_npy, tfid_train_features)
    np.save(outdir + '/' + val_features_npy, tfid_val_features)
    np.save(outdir + '/' + testing_features_npy, tfid_test_features)


    reducer = umap.UMAP(n_components=400)
    training_embedding = reducer.fit_transform(tfid_train_features)
    val_embedding = reducer.transform(tfid_val_features)
    testing_embedding = reducer.transform(tfid_test_features)

    training_features_npy = 'training_features_umap_400.npy'
    testing_features_npy = 'testing_features_umap_400.npy'
    val_features_npy = 'val_features_umap_400.npy'

    np.save(outdir + '/' + training_features_npy, training_embedding)
    np.save(outdir + '/' + val_features_npy, val_embedding)
    np.save(outdir + '/' + testing_features_npy, testing_embedding)

    return training_embedding, val_embedding, testing_embedding


def cli_main():

    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='../data')
    args = parser.parse_args()

    df = read_data()

    training_set, test_set, training_labels, test_labels = train_test_split(df["filtered_content"], df["category"], test_size=0.33)

    training_set, val_set, training_labels, val_labels = train_test_split(training_set, training_labels, test_size=0.2)

    train_embed, val_embed, test_embed = embed(training_set, val_set, test_set, args.outdir)    



if __name__ == '__main__':
    cli_main()
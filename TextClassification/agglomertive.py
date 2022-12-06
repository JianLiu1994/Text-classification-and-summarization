import pandas as pd

from sklearn import preprocessing 
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics, neighbors
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import os
os.chdir('../Preprocessing')
from normalization import normalize_corpus


from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import pickle
from sklearn.cluster import AgglomerativeClustering

np.random.seed(1234)

test_labels = np.load('../data/test_labels.npy')

labels = ['business', 'entertainment', 'health', 'science and technology']

print("AgglomerativeClustering")
for dimension in [400, 800, 1200, 1600, 2000, 3000]:
     print(str(dimension) + ':')

     training_features_pkl = f'training_features_umap_{dimension}.pkl'
     testing_features_pkl = f'test_features_umap_{dimension}.pkl'
     val_features_pkl = f'val_features_umap_{dimension}.pkl'

     training_embedding = pickle.load(open(training_features_pkl, "rb" ) )
     val_embedding = pickle.load(open(val_features_pkl, "rb" ) )
     testing_embedding = pickle.load(open(testing_features_pkl, "rb" ) )

     clustering = AgglomerativeClustering(n_clusters=4, linkage='ward').fit_predict(testing_embedding)
     pred_h = clustering.tolist()

     print(metrics.accuracy_score(test_labels, clustering))
     print(classification_report(test_labels, clustering, target_names=labels))


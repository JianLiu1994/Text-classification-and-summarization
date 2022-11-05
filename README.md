# Text-classification for news and covid dataset

## Python environment (use anaconda3 to install the environment):
```
create -n 8735 python=3.6
conda activate 8735
conda install pandas
pip install newspaper3k or conda install newspaper3k 
pip install nltk
pip install sklearn matplotlib seaborn
```

## Data

### Download news dataset (done):

```
cd Data_collection/workdir
python collect.py 
```

### Combine news dataset (the result file is too large, need to run the scripts to create)

```
python combine.py
cp dataset.csv ../
```

Scrapped news articles from urls provided by UCI Machine Learning repository [link](http://archive.ics.uci.edu/ml/datasets/News+Aggregator)  
For scrapping the news articles, ```Newspaper3k``` [library](https://newspaper.readthedocs.io/en/latest/) built in Python was used. The library contains ```nlp()``` method using which *keywords* and *summary* of the news article can be extracted.   
Article's content and summary have been scrapped to create the data for the project. 

## Data preprocessing
Raw text has unwanted characters (\n,\t,$ etc) and contains stop words (a, an, the) which has to removed before generating the vector representation. The following text preprocessing techniques have been used:  
1. Converting to lower case
2. Removal of stop words
3. Tokenize
4. Removing contractions (does'nt -> does not)
5. Stemming/Lemmatization
6. Use TF-IDF to convert the data into vectors
7. Use UMAP to do the demension reduction


## Text classification:
Classifying the news articles into 4 categories namely Health, Business, Entertainment, Technology using the following ML models:  
1. Logistic regression
2. Support Vector Machine
3. Naive Bayes 
4. Random forest
5. K-NN
6. Latent semantic analysis
7. Spectural clustering
8. Deep learning model (to be decided)

### Results

#### Text classification
| S.no | Model | Accuracy in % (BoW)| Accuracy in % (Tf-idf) |
|------|-------|----------|---------------------|
|1. | Logistic regression | 95.2|94.7 |
|2. | SVM |94.8 | 95.2|
|3. | Naive Bayes | 94.69| 94.54|
|4. | Random forest |92.2 | 92.05|
|5. | K-NN |94.3 | 94.59|


#### Text summarization
| S.no | Model | Rouge-1 |
|------|-------|----------|
|1. | Text rank | 59.2 |
|2. | K-means clustering|54.7 |
|3. | Latent semantic analysis | 52.1|


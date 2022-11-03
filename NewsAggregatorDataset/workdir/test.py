import pandas as pd

colnames=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'] 

df = pd.read_csv('../newsCorpora.csv', names=colnames, sep='\t')

import newspaper
from newspaper import Article
import os 
from multiprocessing import Pool

def download_url(inparams):
    url, category = inparams
    # print("Try to download from " + url)
    try:
        article = Article(url)
        article.download()
        article.parse()
        if len(article.text) > 150 and len(article.text) < 6000:
            article.nlp()
            print(article.text)
            return url, article.text, article.summary, category
    except Exception:
        return None, None, None, None
    return None, None, None, None

process_list = []
urls, contents, summarys, categorys = [], [], [], []
counter = 1
for i in range(len(df)):
    process_list.append([df.loc[i, 'URL'], df.loc[i, 'CATEGORY']])
    if i % 10000 == 0 and i > 0:
        pool = Pool(processes=40)
        results = pool.map(download_url, process_list)
        pool.close()
        pool.join()

        for result in results:
            url, content, summary, category = result
            if url is not None:
                urls += [url]
                contents += [content]
                summarys += [summary]
                categorys += [category]

        new_df = pd.DataFrame({'url': urls, 'content': contents, 'summary': summarys, 'category': categorys})
        new_df.to_csv(f'dataset_{counter}.csv')
        print(f"Saving dataset_{counter}.csv")
        process_list = []
        urls, contents, summarys, categorys = [], [], [], []
        counter += 1

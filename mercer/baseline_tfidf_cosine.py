import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity

mypath = '/scratch/lw1582/capstone/unsupervised/'
df = pd.read_csv(mypath + '/trunc_clustering.csv')
comps = pd.read_csv(mypath + '/competitors.csv')

df.description = df.description.astype(str)
vectorizer = TfidfVectorizer(max_df=0.5, max_features=1000, min_df=2, stop_words='english',use_idf=True)
X = vectorizer.fit_transform(df.description)
score = cosine_similarity(X, Y=None)

with open('/scratch/lw1582/capstone/unsupervised/tfidf_cosine.pickle', 'wb') as handle:
    pickle.dump(score, handle)


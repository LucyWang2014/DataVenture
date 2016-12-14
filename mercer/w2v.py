from gensim.models import word2vec
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
import pickle

model = word2vec.Word2Vec.load_word2vec_format("data/GoogleNews-vectors-negative300.bin", binary=True)



mypath = 'data/'
df = pd.read_csv(mypath + 'clustering_data.csv')


flag_first = True
stop = set(stopwords.words('english'))
for desc in df.description.head(10):
    #curr = []
    for word in desc.lower().split():
        
        if word in stop:
            continue
        try:
            vec = (model[word])
        except:
            continue
        try:
            curr = curr + vec
        except:
            curr = np.zeros((300,), dtype=np.float)
        
    #curr = list(curr/len(curr))
    #curr = [x / len(curr) for x in curr]


    curr = curr/len(curr)


    if flag_first:
        vec = curr
    else:
        vec = np.vstack((vec,curr))
    
    #vec.append(list(curr))

    flag_first = False



score = cosine_similarity(vec, Y=None, dense_output=False)

print ("Shape of score is :" ,score.shape)

with open('/scratch/sdr375/capstone/word2vec_cosine.pickle', 'wb') as handle:
    pickle.dump(score, handle)

            
        
        

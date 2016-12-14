import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords

from gensim.similarities import WmdSimilarity
from gensim.models import Word2Vec



def main():
	
	model = Word2Vec.load_word2vec_format("data/GoogleNews-vectors-negative300.bin.gz",binary=True)
	print "Similarity between france and spain", model.similarity('france', 'spain')
	
	mypath = 'data/'
	df = pd.read_csv(mypath + 'clustering_data.csv')
	#comps = pd.read_csv(mypath + 'competitors.csv')

	sentences = [x.split() for x in df.description.head(10000)]
	

	most_similars_wmd_ng20_top20=[]

	wmd_similarity_top20 = WmdSimilarity(sentences, model, num_best=10)
	for i in range(len(sentences)):
		
		most_similars_wmd_ng20_top20.append(wmd_similarity_top20[sentences[i]])
	print(most_similars_wmd_ng20_top20)



if __name__ == '__main__':
	main()


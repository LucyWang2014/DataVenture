import numpy as np
import pandas as pd
import pickle
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.models import model_from_json

def baseline_model(embedding_vector_length = 64, max_desc_length = 100, top_words = 10000, y_dim = 543):
	# create model
	model = Sequential()
	model.add(Embedding(top_words, embedding_vector_length, input_length=max_desc_length))
	model.add(LSTM(128,return_sequences=True,dropout_W=0.5, dropout_U=0.0))
	model.add(LSTM(64,dropout_W=0.25, dropout_U=0.0))
	model.add(Dense(y_dim, init='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
	return model

def get_data(x, y, size = 500):
	if size > 0:
	    # The test set is sorted by size, but we want to keep random
	    # size example.  So we must select a random selection of the
	    # examples.
	    idx = np.arange(len(x))
	    np.random.shuffle(idx)
	    idx = idx[:size]
	    train_x = [x[n] for n in idx]
	    train_y = [y[n] for n in idx]

	else:
		train_x = x
		train_y = y

	return np.array(train_x), np.array(train_y)
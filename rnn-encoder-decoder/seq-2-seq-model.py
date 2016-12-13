import numpy as np
import pandas as pd
import pickle
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import RepeatVector, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, Adadelta
from keras.models import model_from_json


def pad(x, length = 100):
	# truncate and pad input sequences
	max_desc_length = length
	train_x = sequence.pad_sequences(x, maxlen=max_desc_length)

	return train_x

def transform_data(x, top_words = 10000):
	dim = x.shape
	transformed = np.zeros((dim[0],dim[1],top_words+1))
	for i,seq in enumerate(x):
		for j,word in enumerate(seq):
			transformed[i][j][word] = 1

	return transformed

def seq2seq_model(top_words = 10000, embedding_vector_length = 100, max_desc_length = 100, max_out_length = 15, hidden_neurons = 256):
	
	model = Sequential()  
	model.add(Embedding(top_words, embedding_vector_length, input_length=max_desc_length))
	model.add(LSTM(128,return_sequences = False))
	#model.add(LSTM(input_shape = (max_desc_length,top_words+1), output_dim = hidden_neurons, return_sequences=False))
	model.add(Dense(hidden_neurons, activation="relu"))
	model.add(Dropout(0.2))
	model.add(RepeatVector(max_out_length))
	model.add(LSTM(hidden_neurons, return_sequences=True))
	model.add(TimeDistributedDense(output_dim = max_desc_length, activation = 'softmax'))
	#model.add(Activation("softmax"))  
	model.compile(loss="mean_squared_error", optimizer="adam") 
	print "Model compiled."

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

def main():

	train_data_size = int(sys.argv[1])
	test_data_size = int(sys.argv[2])
	epochs = int(sys.argv[3])

	print 'loading data...'
	mypath = '/Users/Lucy/Google Drive/MSDS/2016Fall/DSGA1006/Data/rnn-encoder-decoder'
	f = open(mypath + '/crunchbase.train.10000.pickle','rb')
	train_x, train_y = pickle.load(f)
	f.close()
	f = open(mypath + '/crunchbase.test.10000.pickle','rb')
	test_x, test_y = pickle.load(f)
	f.close()

	print 'preprocessing data...'
	train_x = pad(train_x, 100)
	test_x = pad(test_x, 100)
	train_y = pad(train_y, 15)
	test_y = pad(test_y, 15)
	#train_x = transform_data(train_x, 10000)
	#test_x = transform_data(test_x, 10000)
	#test_y = transform_data(test_y, 10000)
	#train_y = transform_data(train_y, 10000)


	print 'building model...'
	
	model = seq2seq_model(top_words = 10000, embedding_vector_length = 100, max_desc_length = 100, max_out_length = 15, hidden_neurons = 256)
	print model.summary()

	trunc_x, trunc_y = get_data(train_x, train_y, size = train_data_size)
	trunc_test_x, trunc_test_y = get_data(test_x, test_y, size = test_data_size)

	#training on all data with one lstm layer
	model.fit(trunc_x, trunc_y, nb_epoch=epochs, batch_size=128,verbose = 1, validation_split = 0.1, shuffle = True)

	scores = model.evaluate(trunc_test_x,trunc_test_y, verbose=1)
	print scores

	print 'saving models...'
	# serialize model to JSON
	model_json = model.to_json()
	with open(mypath + "/model.json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(mypath + "/model.h5", overwrite = True)
	json_file.close()

	with open(mypath + '/model_score.csv',"w") as f:
		f.write('model parameters: train_data_size %s, test_data_size %s, epochs %s' % ((train_data_size, test_data_size, epochs)))
		for item in scores:
	  		f.write("%s\n" % item)
	f.close()

	print("Saved model to disk")

if __name__ == '__main__':
	main()





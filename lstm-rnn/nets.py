import numpy as np
import pandas as pd
import pickle
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
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

def encode_y(train_y, test_y):
	# encode class values as integers
	encoder = LabelEncoder()
	encoder.fit(train_y)
	encoded_train_y = encoder.transform(train_y)
	encoded_test_y = encoder.transform(test_y)
	# convert integers to dummy variables (i.e. one hot encoded)
	dummy_train_y = np_utils.to_categorical(encoded_train_y)
	dummy_test_y = np_utils.to_categorical(encoded_test_y)

	return dummy_train_y, dummy_test_y

def pad_x(x, length = 100):
	# truncate and pad input sequences
	max_desc_length = length
	train_x = sequence.pad_sequences(x, maxlen=max_desc_length)

	return train_x

def lstm_model(embedding_vector_length = 256, max_desc_length = 100, top_words = 10000, y_dim = 106):
    # create model
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_desc_length))
    #model.add(LSTM(128,return_sequences=True,dropout_W=0.5, dropout_U=0.0))
    model.add(LSTM(128,dropout_W=0.5, dropout_U=0.0))
    model.add(Dense(y_dim, init='uniform', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

def mlp_model(x_dim = 100, y_dim = 106):
    model = Sequential()
    model.add(Dense(256, input_dim=x_dim, init='uniform'))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(128, init='uniform'))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.25))
    #model.add(Dense(128, init='uniform'))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(y_dim, init='uniform'))
    model.add(Activation('softmax'))

    #opt = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #opt = SGD(lr=10, decay=1e-6, momentum=0.9, nesterov=True)
    opt = Adadelta(lr=5.0, rho=0.95, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
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
	mypath = '/Users/Lucy/Google Drive/MSDS/2016Fall/DSGA1006/Data/lstm-rnn'
	f = open(mypath + '/crunchbase.train.10000.pickle','rb')
	train_x, train_y = pickle.load(f)
	f.close()
	f = open(mypath + '/crunchbase.test.10000.pickle','rb')
	test_x, test_y = pickle.load(f)
	f.close()

	print 'preprocessing data...'
	dummy_train_y, dummy_test_y = encode_y(train_y, test_y)
	train_x = pad_x(train_x, 100)
	test_x = pad_x(test_x, 100)

	print 'building model...'
	estimator = KerasClassifier(build_fn=mlp_model, nb_epoch=200, batch_size=5, verbose=0)
	kfold = KFold(n_splits=10, shuffle=True)
	
	model = mlp_model(x_dim = 100, y_dim = 106)
	print model.summary()

	trunc_x, trunc_y = get_data(train_x, dummy_train_y, size = train_data_size)
	trunc_test_x, trunc_test_y = get_data(test_x, dummy_test_y, size = test_data_size)

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
	model.save_weights(mypath + "/model.h5")
	json_file.close()

	with open(mypath + '/model_score.csv',"w") as f:
		f.write('model parameters: train_data_size %s, test_data_size %s, epochs %s' % ((train_data_size, test_data_size, epochs)))
		for item in scores:
	  		f.write("%s\n" % item)
	f.close()

	print("Saved model to disk")

if __name__ == '__main__':
	main()





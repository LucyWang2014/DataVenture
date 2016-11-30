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
from utils import baseline_model, get_data
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
#from sklearn.preprocessing import LabelEncoder
#from sklearn.pipeline import Pipeline

'''
load these packages for the code to run:
module load keras/1.1.1
module load scipy/intel/0.18.0
module load tensorflow/python2.7/20160721
'''

print 'start training...'
train_data_size = int(sys.argv[1])
test_data_size = int(sys.argv[2])
epochs = int(sys.argv[3])

#load train and test data
mypath = '/scratch/lw1582/capstone/lstm-rnn'
f = open(mypath + '/crunchbase.train.10000.pickle','rb')
train_x, dummy_train_y = pickle.load(f)
f.close()
f = open(mypath + '/crunchbase.test.10000.pickle','rb')
test_x, dummy_test_y = pickle.load(f)
f.close()

x_length = map(len, train_x)
print 'max description length is %s. min description length is %s' % (max(x_length), min(x_length))
print 'average description length is %s. median description length is %s' % (np.mean(x_length), np.median(x_length))

'''
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(train_y)
encoded_train_y = encoder.transform(train_y)
encoded_test_y = encoder.transform(test_y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_train_y = np_utils.to_categorical(encoded_train_y)
dummy_test_y = np_utils.to_categorical(encoded_test_y)

# truncate and pad input sequences
max_desc_length = 100
train_x = sequence.pad_sequences(train_x, maxlen=max_desc_length)
test_x = sequence.pad_sequences(test_x, maxlen=max_desc_length)

#create estimator and kfolds for cross validation
#estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
#kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

'''

model = baseline_model(embedding_vector_length = 64, max_desc_length = 100, top_words = 10001, y_dim = 543)
model.summary()

trunc_x, trunc_y = get_data(train_x, dummy_train_y, size = train_data_size)
trunc_test_x, trunc_test_y = get_data(test_x, dummy_test_y, size = test_data_size)

model.fit(trunc_x, trunc_y, nb_epoch=epochs, batch_size=32,verbose = 1, validation_split = 0.1, shuffle = True)

scores = model.evaluate(trunc_test_x,trunc_test_y, verbose=1)
print scores

# serialize model to JSON
model_json = model.to_json()
with open(mypath + "model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
json_file.close()

with open(mypath + 'model_score.csv',"w") as f:
	f.write('model parameters: train_data_size %s, test_data_size %s, epochs %s' % ((train_data_size, test_data_size, epochs)))
	for item in scores:
  		f.write("%s\n" % item)
f.close()

print("Saved model to disk")

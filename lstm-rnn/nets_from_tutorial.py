'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
import pickle

max_words = 100
batch_size = 128
nb_epoch = 100

print('Loading data...')
mypath = '/Users/Lucy/Google Drive/MSDS/2016Fall/DSGA1006/Data/lstm-rnn'
f = open(mypath + '/crunchbase.train.5000.pickle','rb')
X_train, y_train = pickle.load(f)
f.close()
f = open(mypath + '/crunchbase.test.5000.pickle','rb')
X_test, y_test = pickle.load(f)
f.close()
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Vectorizing sequence data...')
max_desc_length = max_words
X_train = sequence.pad_sequences(X_train, maxlen=max_desc_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_desc_length)
#tokenizer = Tokenizer(nb_words=max_words)
#X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
#X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_train_y = encoder.transform(y_train)
encoded_test_y = encoder.transform(y_test)
# convert integers to dummy variables (i.e. one hot encoded)
Y_train = np_utils.to_categorical(encoded_train_y)
Y_test = np_utils.to_categorical(encoded_test_y)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

nb_classes = np.max(encoded_train_y)+1
print(nb_classes, 'classes')

print('Building model...')
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=1, validation_split=0.1)
score = model.evaluate(X_test, Y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
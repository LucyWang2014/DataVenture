"""
This files contains the functions used to preprocess the dataset for lstm-rnn model

created by: Lucy Wang
11.27.16
"""

import pandas as pd
import numpy as np
import pickle
import itertools
import nltk
from nltk.corpus import stopwords
import unicodedata
import sys

sentence_start_token = "SENTENCESTART"
sentence_end_token = "SENTENCEEND"


def tokenize(sentences):

    def strip_punctuation(text):
        """
        >>> strip_punctuation(u'something')
        u'something'

        >>> strip_punctuation(u'something.,:else really')
        u'somethingelse really'
        """
        punctutation_cats = set(['Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po'])
        return ''.join(x for x in text if unicodedata.category(x) not in punctutation_cats)

    print 'Tokenizing...',
    print 'Splitting full descriptions into sentences...'
    sentences = itertools.chain(*[[nltk.sent_tokenize(x.decode('utf-8').lower())] for x in sentences])
    mod_sentences = []
    for desc in sentences:
        full_desc = ""
        for x in desc:
            s = "%s %s %s " % (sentence_start_token, x, sentence_end_token)
            full_desc = full_desc + s
        mod_sentences.append(full_desc)
    
    mod_sentences = map(strip_punctuation, mod_sentences)
    print 'Tokenizing sentences..,'
    toks = [nltk.word_tokenize(sent) for sent in mod_sentences]
    print 'Done'

    return toks


def build_dict(descriptions):

    tokenized_sentences = tokenize(descriptions)

    # Count the word frequencies
    print 'Building dictionary..'
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    word_freq = sorted(word_freq.items(), key=lambda i: i[1], reverse=True)
    print "Found %d unique words tokens." % len(word_freq.items())

    counts = [x[1] for x in word_freq]
    keys = [x[0] for x in word_freq]

    worddict = dict()

    idx = 0

    worddict[sentence_start_token] = 2
    worddict[sentence_end_token] = 3
    for word in keys:
        if word != sentence_start_token and word != sentence_end_token:
            worddict[word] = idx + 4 # leave 0, 1 (UNK), 2 (sentence start), and 3(sentence end)
            idx += 1 
        
    print numpy.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict


def grab_data(sentences, dictionary):

    sentences = data.clean_description

    sentences = tokenize(sentences)

    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words[:-1]]


    return seqs

def main():

    vocab_size = sys.argv[1]

    mypath = '/Users/Lucy/Google Drive/MSDS/2016Fall/DSGA1006/Data'
    f = open(mypath + '/lstm-rnn/lstm_data.pickle','rb')
    X_train, X_test, y_train, y_test = pickle.load(f)
    f.close()

    train_desc = X_train.clean_description
    test_desc = X_test.clean_description

    worddict = build_dict(train_desc)

    keys = worddict.keys()
    values = worddict.values()

    dictionary = {keys[values.index(v)]: v for v in sorted(word_to_index.values())[:vocab_size]}


    train_x = grab_data(train_desc, dictionary)
    test_x = grab_data(test_desc, dictionary)

    f = open(fdataset_path + '../../Data/lstm-rnn/crunchbase.train.pickle', 'wb')
    pkl.dump((train_x, train_y), f, -1)
    f.close()

    f = open(fdataset_path + '../../Data/lstm-rnn/crunchbase.test.pickle', 'wb')
    pkl.dump((test_x, test_y), f, -1)
    f.close()

    f = open(fdataset_path + '../../Data/lstm-rnn/crunchbase.dict.pickle', 'wb')
    pkl.dump(dictionary, f, -1)
    f.close()

if __name__ == '__main__':
    main()

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

'''
load these packages for the code to run:
module load anaconda/2.3.0
'''

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
    print 'Tokenizing sentences...'
    toks = [nltk.word_tokenize(sent) for sent in mod_sentences]
    print 'Done'

    return toks


def build_dict(descriptions, vocab_size):

    tokenized_sentences = tokenize(descriptions)

    # Count the word frequencies
    print 'Building dictionary...'
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    #word_freq = sorted(word_freq.items(), key=lambda i: i[1], reverse=True)
    word_freq = word_freq.most_common(vocab_size-1)
    #print "Found %d unique words tokens." % len(word_freq)

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
        
    print np.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict


def grab_data(sentences, dictionary):

    sentences = tokenize(sentences)

    seqs = [None] * len(sentences)
    for idx, words in enumerate(sentences):
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words[:-1]]


    return seqs

def main():

    vocab_size = int(sys.argv[1])
    print vocab_size

    mypath = '/Users/Lucy/Google Drive/MSDS/2016Fall/DSGA1006/Data'
    f = open(mypath + '/lstm-rnn/lstm_data.pickle','rb')
    train_x, test_x, train_y, test_y = pickle.load(f)
    f.close()

    train_desc = train_x.clean_description
    test_desc = test_x.clean_description

    dictionary = build_dict(train_desc, vocab_size)

    #keys = worddict.keys()
    #values = worddict.values()

    #dictionary = {keys[values.index(v)]: v for v in sorted(values)[:vocab_size-1]}
    print "Using vocabulary size %d." % vocab_size


    train_x = grab_data(train_desc, dictionary)
    test_x = grab_data(test_desc, dictionary)

    print "saving new data files..."
    f = open('../../Data/lstm-rnn/crunchbase.train.%s.pickle' % vocab_size, 'wb')
    pickle.dump((train_x, train_y), f, -1)
    f.close()

    f = open('../../Data/lstm-rnn/crunchbase.test.%s.pickle' % vocab_size, 'wb')
    pickle.dump((test_x, test_y), f, -1)
    f.close()

    f = open('../../Data/lstm-rnn/crunchbase.dict.%s.pickle' % vocab_size, 'wb')
    pickle.dump(worddict, f, -1)
    f.close()

if __name__ == '__main__':
    main()

__author__ = 'yonglin'
from __future__ import  division
import nltk
from nltk.corpus import movie_reviews
import collections
import nltk.metrics
from nltk.corpus import stopwords
from nltk.stem import *
from nltk.stem.porter import *

import random
import re
import pickle

## documents is a nested list contains tuples. And each tuple has two elements:
## Element_1 is a list wich contains the tokens of description of a movie
## Element_2 is a string which stands for the 'negative' or 'positive'
## e.g for any element movie_1 in documents, it might be like this
## movie_1 = (['love', ':', 'romantic', 'NY'], 'neg')
documents = [(list(movie_reviews.words(fileid)), category)
			for category in movie_reviews.categories()
			for fileid in movie_reviews.fileids(category)]

  ## movie_reviews.fileids('neg') will return a list which contain all the fileID of the negative movies
  	## it looks like ['neg/cv997_5152.txt', 'neg/cv998_15691.txt', 'neg/cv999_14636.txt']
  ## movie_reviews.categories() will return a list ['pos', 'neg']
  ## movie_reviews.words(fileID) will return a list of tokens of movie fileID's description

random.shuffle(documents)
  ## make cocuments disordered randomly
doc_file = 'documents.data'
f = open(doc_file, 'wb')
pickle.dump(documents, f)
f.close()
  ## sotore the training and testing set permanently by Pickle

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
  ## movie_reviews.words() contains all the tokens appear in all the movie descriptions, which has 1583820 tokens
  ## at fisrt we make all the tokens as lowercase
  ## nltk.FreqDist(['love', 'love', 'romantic', 'NY']) will return a generic Python Dictionary which has descend order by the value
  ## {'love': 2, 'romantic': 1, 'NY': 1}
  ## so all_words here is also a generic Python Dicionary

word_features = all_words.keys()[:1000]
  ## we just needs the fisrt 1000 frequent tokens

def document_features(document): 
	'''input a list of tokens
	   output a Dicionary which is like {'contains(love)': True, 'contains(Cats)': False}'''
	document_words = set(document) 
      ## transform the list to be a Python set

	features = dict()
 	for word in word_features:
 		features['contains(%s)' % word] = (word in document_words)
          ## judge whether a word is in the first 1000 frequent token set
	return features

featuresets = [(document_features(d), c) for (d,c) in documents]
  ## feature is a nested Python list which is like [({}, category)]
  ## the dictionary in it is the feature dictionary of a that movie's description

train_set = featuresets[:1600]
test_set = featuresets[1600:]
  ## split the total set into training and testing sets



classifier = nltk.NaiveBayesClassifier.train(train_set)
  ## train a classifier by using the train_set
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)
 
print 'pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos'])
print 'pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos'])
print 'pos F-measure:', nltk.metrics.f_measure(refsets['pos'], testsets['pos'])
print 'neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg'])
print 'neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg'])
print 'neg F-measure:', nltk.metrics.f_measure(refsets['neg'], testsets['neg'])

print 'accuracy:', nltk.classify.accuracy(classifier, test_set)
#classifier.show_most_informative_features(5)



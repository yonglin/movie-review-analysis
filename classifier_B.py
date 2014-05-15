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

doc_file = 'documents.data'
f = open(doc_file, 'rb')
documents = pickle.load(f) # load the object from the file
f.close()
  ## we use Pickle to get back the documents we have used before

tokens = movie_reviews.words()

def wash_data(desc):
  '''input a list of strings
  output a list of washed strings'''

  tokens = ' '.join(desc)
  #tokens = re.sub(r'[^a-zA-Z ]','',tokens)
    ## just keeping the alphabet might improve the result
  tokens = re.sub(r'[^a-zA-Z0-9 ]','',tokens) #removing non alpha-numeric characters might improve the result
  tokens = nltk.word_tokenize(tokens)

  stopwords = nltk.corpus.stopwords.words('english')
   ## remove the stopwords
  tokens = [token.lower() for token in tokens if token.lower() not in stopwords]
  #tokens = [token.lower() for token in tokens]
  #stemmer = PorterStemmer()
  #tokens = [stemmer.stem(token) for token in tokens]
   ## stem each word, it seems if we do not stem the words, we will get a better result

  return tokens

tokens = wash_data(tokens)

all_words = nltk.FreqDist(tokens)
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
  ## featuresets is a nested Python list which is like [({}, category)]
  ## the dictionary in it is the feature dictionary of a that movie's description

train_set, test_set = featuresets[:1600], featuresets[1600:]
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



 

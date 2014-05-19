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

from numpy import percentile

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm


doc_file = 'documents.data'
f = open(doc_file, 'rb')
documents = pickle.load(f) # load the object from the file
f.close()
  ## we use Pickle to get back the documents we have used before

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

def wash_join(desc):
  ## transform a string list into a string
	tokens = wash_data(desc)
	tokens = ' '.join(tokens)
	return tokens

###########
desc = [(wash_join(item[0]), item[1]) for item in documents]
corpus_moive = [d for (d,c) in desc]
  ## CountVectorizer can autmatically deal with the counting stuff 
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus_moive)
############

########## change 'pos' and 'neg' into 1 and 0, respectively
y = [item[1] for item in desc]
Y = list()
for item in y:
	if item == 'pos':
		Y.append(1)
	else:
		Y.append(0)
#http://scikit-learn.org/stable/modules/feature_extraction.html

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)
  ## get the tfidf matrix
X_train = tfidf[0:1600,]
y_train = Y[0:1600]
#test_set = L1[0:1600]
X_test= tfidf[1600:,]
y_test = Y[1600:]
#y_train = 


#http://scikit-learn.org/stable/modules/svm.html

C = 1  # SVM regularization parameter
#clf = svm.SVC()
#clf.fit(X_train, y_train) 

svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train) 
  ## linear kernel is not so~~~~o good
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train) 
  ## linear kernel can get a good result
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train) 
  ## poly kernel is not so~~~~o good
lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train) 
  ## LinearSVC  can get the best result

  ## train a classifier by using the train_set
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)


for i, item in enumerate(y_test):
    refsets[item].add(i)

    observed = lin_svc.predict(X_test[i,])
    testsets[observed[0]].add(i)

print 'Results of lin_svc:\n' 
print 'lin_svc pos precision:', nltk.metrics.precision(refsets[1], testsets[1])
print 'lin_svc pos recall:', nltk.metrics.recall(refsets[1], testsets[1])
print 'lin_svc pos F-measure:', nltk.metrics.f_measure(refsets[1], testsets[1])
print 'lin_svc neg precision:', nltk.metrics.precision(refsets[0], testsets[0])
print 'lin_svc neg recall:', nltk.metrics.recall(refsets[0], testsets[0])
print 'lin_svc neg F-measure:', nltk.metrics.f_measure(refsets[0], testsets[0])
print 'lin_svc accuracy:', nltk.metrics.scores.accuracy(refsets, testsets)
print '\n'

####################
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)


for i, item in enumerate(y_test):
    refsets[item].add(i)

    observed = rbf_svc.predict(X_test[i,])
    testsets[observed[0]].add(i)

print 'Results of rbf_svc:\n'  
print 'rbf_svc pos precision:', nltk.metrics.precision(refsets[1], testsets[1])
print 'rbf_svc pos recall:', nltk.metrics.recall(refsets[1], testsets[1])
print 'rbf_svc pos F-measure:', nltk.metrics.f_measure(refsets[1], testsets[1])
print 'rbf_svc neg precision:', nltk.metrics.precision(refsets[0], testsets[0])
print 'rbf_svc neg recall:', nltk.metrics.recall(refsets[0], testsets[0])
print 'rbf_svc neg F-measure:', nltk.metrics.f_measure(refsets[0], testsets[0])
print 'rbf_svc accuracy:', nltk.metrics.scores.accuracy(refsets, testsets)
print '\n'
###############

####################
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)


for i, item in enumerate(y_test):
    refsets[item].add(i)

    observed = svc.predict(X_test[i,])
    testsets[observed[0]].add(i)

print 'Results of svc:\n'  
print 'svc pos precision:', nltk.metrics.precision(refsets[1], testsets[1])
print 'svc pos recall:', nltk.metrics.recall(refsets[1], testsets[1])
print 'svc pos F-measure:', nltk.metrics.f_measure(refsets[1], testsets[1])
print 'svc neg precision:', nltk.metrics.precision(refsets[0], testsets[0])
print 'svc neg recall:', nltk.metrics.recall(refsets[0], testsets[0])
print 'svc neg F-measure:', nltk.metrics.f_measure(refsets[0], testsets[0])
print 'svc accuracy:', nltk.metrics.scores.accuracy(refsets, testsets)
print '\n'
###############

####################
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)


for i, item in enumerate(y_test):
    refsets[item].add(i)

    observed = poly_svc.predict(X_test[i,])
    testsets[observed[0]].add(i)

print 'Results of poly_svc:\n'  
print 'poly_svc pos precision:', nltk.metrics.precision(refsets[1], testsets[1])
print 'poly_svc pos recall:', nltk.metrics.recall(refsets[1], testsets[1])
print 'poly_svc pos F-measure:', nltk.metrics.f_measure(refsets[1], testsets[1])
print 'poly_svc neg precision:', nltk.metrics.precision(refsets[0], testsets[0])
print 'poly_svc neg recall:', nltk.metrics.recall(refsets[0], testsets[0])
print 'poly_svc neg F-measure:', nltk.metrics.f_measure(refsets[0], testsets[0])
print 'poly_svc accuracy:', nltk.metrics.scores.accuracy(refsets, testsets)
print '\n'
###############
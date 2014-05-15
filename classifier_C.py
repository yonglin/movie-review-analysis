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

def lexical_diversity(text):
	'''output the lexical_diversity of a list'''
	return len(text) / len(set(text))

def get_threshold_diversity(documents):
	lexi_div_documents = [lexical_diversity(item[0]) for item in documents]
	  ## the lexical_diversity of all movive descriptions
	threshold_div = percentile(lexi_div_documents, 50)
	  ## percentile is a function from numpy which can return the percentile of a ummber list
	  ## here we will use this 50% percentile(median) to transform real value into binary bits
	return threshold_div

def get_lexi_feature(document):
	'''we just consider the document who has less than threshold_div as a True Value'''
	threshold_div = get_threshold_diversity(documents)
	features = dict()
	features['lexi_div>(%s)' % threshold_div] = (lexical_diversity(document) < threshold_div)

	return features

def get_threshold_length(documents):
	len_documents = [len(item[0]) for item in documents]
	  ## the lexical_diversity of all movive descriptions
	threshold_len = [percentile(len_documents, p) for p in [25, 50, 75] ]
	  ## percentile is a function from numpy which can return the percentile of a ummber list
	  ## here we will use this 25%, 50%, and 75% percentiles to transform real value into binary bits
	return threshold_len

def get_length_feature(document):
	'''we just consider the document who has less than threshold_div as a True Value'''
	threshold_len = get_threshold_length(documents)
	len_doc = len(document)
	features = dict()
	features['0-25'] = (len_doc <= threshold_len[0])
	features['25-50'] = (threshold_len[0] < len_doc <= threshold_len[1])
	features['50-75'] = (threshold_len[1] < len_doc <= threshold_len[2])
	features['75-100'] = (len_doc > threshold_len[2])

	return features

def get_count_feature(document, N = 1):
	'''we just consider the word who occurs more than once as a True Value'''
	document_words = set(document) 

	features = dict()
	for word in word_features:
		features['more_than_N(%s)' % word] = (document.count(word) > N)

	return features

 
def document_features(document): 
	'''input a list of tokens
	   output a Dicionary which is like {'contains(love)': True, 'contains(Cats)': False}'''

	lexi_feature = get_lexi_feature(document)
	len_feature = get_length_feature(document)
	count_feature = get_count_feature(document)
      ## transform the list to be a Python set

	features = dict(count_feature.items() + lexi_feature.items() + len_feature.items())

	return features
featuresets = list()
for i, (d,c) in enumerate(documents):
	print i+1
	featuresets.append((document_features(d), c))

#featuresets = [(document_features(d), c) for (d,c) in documents]
  ## featuresets is a nested Python list which is like [({}, category)]
  ## the dictionary in it is the feature dictionary of a that movie's description

train_set, test_set = featuresets[:1600], featuresets[1600:]
  ## split the total set into training and testing sets

classifier = nltk.NaiveBayesClassifier.train(train_set)
  ## train a classifier by using the train_set

##### I referred the following codes from the below link ###########################################
# http://streamhacker.com/2010/05/24/text-classification-sentiment-analysis-stopwords-collocations/#

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

##### I referred the above codes from the below link ###############################################
# http://streamhacker.com/2010/05/24/text-classification-sentiment-analysis-stopwords-collocations/#


print 'accuracy:', nltk.classify.accuracy(classifier, test_set)
#classifier.show_most_informative_features(5)



 



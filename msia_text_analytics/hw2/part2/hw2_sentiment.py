# -*- coding: utf-8 -*-
"""
MSiA - Text Analytics - Homework 2
Author: Luis Steven Lin
IDE: Spyder, Python 2

Naive Bayes Immplementation

# References
# http://www.nltk.org/book/ch03.html
# http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
# http://stackoverflow.com/questions/7152762/how-to-redirect-print-output-to-a-file-using-python
# http://stackoverflow.com/questions/2225564/get-a-filtered-list-of-files-in-a-directory
# http://matthewrocklin.com/blog/work/2014/05/01/Fast-Data-Structures/
# http://pandas.pydata.org/pandas-docs/stable/dsintro.html
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html

"""

#%% setup  ##################################################################

import os
import re
import sys
from __future__ import division # division results in float if both integers
import re
import math
import pandas
import numpy as np
import time
import matplotlib.pyplot as plt


#%% Class ##################################################################

class NaiveBayes:
    """ Naive Bayes model
    Args:
        d_class (dict): key: class, value = (dict): key= document number, value= list of words
    """
    def __init__(self, d_class):
        self.d_class = d_class
        self.prob = None
        self.log_prob = None
        self.train_docs = None
        self.vocabulary = None
        self.log_priors = None
        self.priors = None
        self.predictions = None
        self.recall = None
        self.precision = None
        self.crosstab = None
        self.F = None
        
        
    def getCounts(self, d_doc, keys):
        """ Gets the frequencies for elements across dictionary with the input keys
        
        args:
            d_doc (dict): key= document number, value= list of words
            keys (list or set): only consider this keys
        
        returns:
            pandas.core.series.Series: key = element, value = count
        """
        
        # flatten so list of all elements for all lists from the dictionary
        elements_unique = [v for k in d_doc for v in d_doc[k] if k in keys]
        series = pandas.Series(elements_unique)
        c = series.value_counts()
        return c
     
     
    def computeConditional(self, d_class, train_docs):
        """ Computes the conditinoal probability words given a class
        
        args:
            d_class (dict): key: class, value = (dict): key= document number, value= list of words
            train_docs (dict): key: class (pos or neg), value = list of documents IDs
                               used for training
        
        returns:
            dict: key = class (pos or neg), value: pandas.core.series.Series
                  with probabilities as values and words as index
        
        """
        # key = pos or neg, value =  pandas series with index=words, value=count  
        counts = {k: self.getCounts(d_class[k],train_docs[k]) for k in d_class}
            
        # Vocabulary = distinct words from all classes
        vocabulary = set([v for k in counts for v in counts[k].index])
        
        # Vocabulary size
        V = len(vocabulary)
        
        # for each class, fill missing values for words of that class that are not in the vocabulary
        # do this by creating a pandas series from the current count series using vocabulary
        # as index. The resulting series will include all the words in the vocabulary as index
        # with value equal to the count in the orignal series, except for words that are not
        # in the original series which will get a NaN value, which in turns gets filled with zero
        counts_fill = {k: pandas.Series(v, index = vocabulary).fillna(0, inplace=False) for k,v in counts.items()}
        
        # check 
        key_test = counts.keys()[0]
        assert len(counts[key_test]) < len(counts_fill[key_test]) # since now had words of vocabulary
        assert sum(counts[key_test]) == sum(counts_fill[key_test]) # since value for added words = 0
        assert V == counts_fill[key_test].size
        
        # key: pos or neg, value = number of words in that class
        n_words = {k: sum(v) for k,v in counts_fill.items()}
        
        # probabilty of word given class = (count(word in class) + 1 )/( total number of words in class) + V )
        prob = {k: (v+1)/(n_words[k]+V) for k,v in counts_fill.items()}
        
        return prob
        
    
    def trainModel(self, train_docs ):
        """ Computes the conditinoal probability words given a class
        
        args:
            d_class (dict): key: class, value = (dict): key= document number, value= list of words
            train_docs (dict): key: class (pos or neg), value = list of documents IDs
                               used for training                                            
        """
        # number of training documents
        N = sum(len(v) for v in train_docs.values())
        priors = {k: len(v)/N for (k,v) in train_docs.items()}
        log_priors = {k: math.log(v) for (k,v) in priors.items()}
        
        prob = self.computeConditional(self.d_class, train_docs)
        log_prob = {k: np.log(v, dtype='float64') for (k,v) in prob.items()}
        
        # set class varaibles
        self.train_docs = train_docs    
        self.priors = priors
        self.log_priors = log_priors
        self.prob = prob
        self.log_prob = log_prob
        self.vocabulary = prob[prob.keys()[0]].index # same vocabulary so choose any class to get vocabulary
        

    def classify(self,list_doc):
        """ Classify document to a class
        
        args:
            list_doc = list of words
        
        returns:
            string or int : classification
        
        """
        
        series = pandas.Series(list_doc)
        pred_count = series.value_counts()
        
        # multiply common elements from counts of vector to predict with probabiliy vector
        # (elements not in common will be filled with value zero). Then sum all values
        # also add the prior
        pred = {k: self.log_priors[k] + 
                   sum(pandas.Series.multiply(pred_count,v).fillna(0,inplace=False)) for (k,v) in self.log_prob.items()}
        
        return max(pred, key=pred.get)
        
        
    def predictDocuments(self, test_docs):
        """ Classify test documents
                
        args:
            test_docs (dict): key: class (pos or neg), value = list of documents IDs
                               used for testing
        
        returns:
            pandas dataframe: rows = docID, columns = actual and predicted classes
        
        """
        
        # tuple = (docID, actual, prediction)
        results = [(doc, k, self.classify(self.d_class[k][doc])) for (k,doc_list) in test_docs.items() for doc in doc_list]
        
        results_dic = {"actual": None, "predicted": None}
        results_dic["actual"] = [t[1] for t in results]
        results_dic["predicted"] = [t[2] for t in results]
        
        # dataframe
        df = pandas.DataFrame(results_dic, index = [t[0] for t in results])  
        df.index.name = "docID"
        
        self.predictions = df
        
        
    def evaluate(self):
        """ Evaluate test by computing precision, recall, F
        
        """
        comb = [(x,y) for x in ["pos","neg"] for y in ["pos","neg"]]
        classification = ["tp", "fn", "fp", "tn"]
        
        classif_dict = {k: v for (k,v) in zip(classification, comb)}
        classif_counts = {k:  sum((self.predictions["actual"] == v[0])*(self.predictions["predicted"] == v[1]))
                                    for (k,v) in classif_dict.items() }
        
        crosstab = pandas.crosstab(self.predictions["actual"], self.predictions["predicted"])
        
        precision = classif_counts["tp"]/(classif_counts["tp"] + classif_counts["fp"])
        recall = classif_counts["tp"]/(classif_counts["tp"] + classif_counts["fn"])
        
        beta = 1

        F = (beta**2 + 1)*precision*recall/((beta**2)*precision + recall)
        
        self.recall = recall
        self.precision = precision
        self.crosstab = crosstab
        self.F = F
        
    def getPreditions(self):
        return self.predictions
        
    def getEvaluation(self):
        return {"recall": self.recall, "precision":self.precision, 
                "crosstab": self.crosstab, "F": self.F}
                

#%% Load Data  ##############################################################

path = '/Users/Steven/Documents/Academics/3_Graduate School/2014-2015_NU/MSIA_490_Text_Mining/hw/hw2'
os.chdir(path)

# strucure: assume each folder represents a class, and each folder contains documents
# get classes (e.g. pos or negative), ignore folders with .DS_Store
classes = os.listdir(os.path.join(path,"review_polarity", "txt_sentoken"))
classes = [c for c in classes if re.match(r'[^\.]',c)]
classes.sort() # sort alphabetically in increasing order ['neg', 'pos']
         
# key = pos or neg, value =  dictionaries with key = filenumber, value = list tokens       
reviews = {k: {} for k in classes}

for polarity in reviews:
    
    sentiment_path = os.path.join(path,"review_polarity", "txt_sentoken",polarity)
    
    fileNames = [f for f in os.listdir(sentiment_path ) if re.match(r'.+\.txt', f)]
    
    for fileName in fileNames:
        # read the file as string
        in_file_name = os.path.join(sentiment_path ,fileName)
        f = open(in_file_name)
        text = f.read()
        
        # split by new line (takes care of headers before bios)
        text_split = text.split('\n')
        
        # sentence segmentation of each line (each element is a list of sentences)
        sentences = [line.split(" ") for line in text_split if line != '']
        # sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        
        # flatten nested list so each element is a token
        tokens = [token for sentence in sentences for token in sentence if token!='']
        
        # 'cv000_29590.txt' --> 0
        fileNumber = int(re.search(r'[0-9]+',fileName.split("_")[0]).group(0))
        
        reviews[polarity][fileNumber] = tokens
    
        f.close()   
            
        
#%% Train and Test sample ####################################################
        
# training document list in alphabetical order (same order as classes)  
train_docs = [range(2), range(2)]
# key: class, value: list of documents
train_docs = {k: v for k,v in zip(classes,train_docs)}     

NB = NaiveBayes(reviews)        
NB.trainModel(train_docs)

# test document list in alphabetical order (same order as classes)  
test_docs = [range(2,5), range(2,5)]
# key: class, value: list of documents
test_docs = {k: v for k,v in zip(classes,test_docs)}   

NB.predictDocuments(test_docs)
NB.evaluate()
NB.getEvaluation()
NB.getPreditions()


#%% Function to run various NB different training set #######################

def runNB(sizeTrain, testDoc, d_class):
    """ Run Naive Bayes
     
    args:
        d_class (dict): key: class, value = (dict): key= document number, value= list of words
        sizeTrain (list): list of size docuemnts train (e.g. 800 means first 8oo documents)
        testDoc (list): documents to test
        
    reutrns:
        dict: key = size, value = dict with key = F, crosstab, precision, recall, predictions
    
    """
    
    results = {k: None for k in sizeTrain}

    t0 = time.time()
    
    for size in sizeTrain:
        
        # training document list in alphabetical order (same order as classes)  
        train_docs = [range(size), range(size)]
        # key: class, value: list of documents
        train_docs = {k: v for k,v in zip(classes,train_docs)}     
        
        NB = NaiveBayes(d_class)        
        NB.trainModel(train_docs)
        
        # test document list in alphabetical order (same order as classes)  
        test_docs = [testDoc , testDoc ]
        # key: class, value: list of documents
        test_docs = {k: v for k,v in zip(classes,test_docs)}   
        
        NB.predictDocuments(test_docs)
        NB.evaluate()
    
        results[size] = NB.getEvaluation()
        results[size]["predictions"] = NB.getPreditions()
        
        print("Size: {0}, Time so far: {1}".format(size, round(time.time()-t0,2)))
    
    return results
    
#%% Function to covert results dictionary to dataframe ######################
    
def resultsToPandas(results):
    """ Function to covert results dictionary to dataframe
    
    args:
        dict: key = size, value = dict with key = F, crosstab, precision, recall
    
    returns:
        pandas dataframe class > document:  columns = precision, recall , f
    
    """
    
    results_tuples = [(size, v["precision"], v["recall"], v["F"]) for (size,v) in sorted(results.items(),key= lambda x: x[0])]
    results_columns = {}
    results_columns["precision"] = [round(v[1],2) for v in results_tuples  ]
    results_columns["recall"] = [round(v[2],2) for v in results_tuples  ]
    results_columns["F"] = [round(v[3],2) for v in results_tuples  ]
    
    results_df = pandas.DataFrame(results_columns, index = [v[0] for v in results_tuples  ]) 
    results_df.index.name = "docID"
    
    return results_df

    
#%% Train and Test different sizes ###########################################
sizeTrain = [100, 300, 500, 600] # the first 100, the first 300 ,etc
testDoc = range(600,800) # 600 to 799

results = runNB(sizeTrain, testDoc, reviews)
results_df = resultsToPandas(results)
results_df

results_df.to_csv("hw_2_6_performance1_NB.txt", sep=',', encoding='utf-8')    
    

#%% Plotresults ###########################################

colors = {"precision": "b", "recall": "g", "F": "r"}

for column in results_df.columns:
    plt.plot(results_df.index,results_df[column],'-{}o'.format(colors[column]), label=column)

# Place a legend above this legend, expanding itself to
# fully use the given bounding box.
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)

plt.xlabel('size of training set')
plt.ylabel('evluation metrics')
plt.xlim(0, 700)
plt.ylim(0.6,1)
plt.savefig('hw_2_6_performance1_NB.png')
plt.show()


#%% Run on hold out test  ###########################################

sizeTrain = [800]  # the first 100, the first 300 ,etc
testDoc = range(800,1000) # 600 to 799

results = runNB(sizeTrain, testDoc, reviews)
results_df = resultsToPandas(results)
results_df

results_df.to_csv("hw_2_6_performance2_NB.txt", sep=',', encoding='utf-8')    

results[800]["predictions"].to_csv("hw_2_6_predictions2_NB.txt", sep=',', encoding='utf-8')


#%% Compare to NB NLTK (code from link below)  ################################

# http://www-rohan.sdsu.edu/~gawron/python_for_ss/course_core/book_draft/text/text_classification.html

# We import Bo Pang and Lillian Lee’s movie reviews corpus [PANGLEE2004], which is one of the NLTK corpora.
from nltk.corpus import movie_reviews as mr

# Use a Naive Bayes Classifier
from nltk.classify import NaiveBayesClassifier

data = dict(pos = mr.fileids('pos'),
            neg = mr.fileids('neg'))
            
print mr.readme()

# The character by character view uses the raw method:
print mr.raw(data['pos'][0])[:100]

# The word by word character view uses the words method:
print mr.words(data['pos'][0])[:10]

def unigram_features (words):
   return dict((word, True) for word in words)

def extract_features (corpus, file_ids, cls, feature_extractor=unigram_features):

   return [(feature_extractor(corpus.words(i)), cls) for i in file_ids]

#### Training

# Use 90% of the data for training
neg_training = extract_features(mr, data['neg'][:900], 'neg',
                                feature_extractor=unigram_features)

# Use 10% for testing the classifier on unseen data.
neg_test = extract_features(mr, data['neg'][900:], 'neg',
                                feature_extractor=unigram_features)

pos_training = extract_features(mr, data['pos'][:900],'pos',
                                feature_extractor=unigram_features)

pos_test = extract_features(mr, data['pos'][900:],'pos',
                                feature_extractor=unigram_features)

train_set = pos_training + neg_training

test_set = pos_test + neg_test

classifier = NaiveBayesClassifier.train(train_set)

#### Classifying

predicted_label0 = classifier.classify(pos_test[0][0])

print 'Predicted: %s Actual: pos' % (predicted_label0,)

predicted_label1 = classifier.classify(neg_test[0][0])

print 'Predicted: %s Actual: neg' % (predicted_label1,)

classifier.classify(unigram_features('Inception is the best movie ever'.split()))
classifier.classify(unigram_features("I don't know how anyone could sit through Inception".split()))

#### Most informative features

#These are the features for which the ratio of the positive to negative probability (or vice versa) is the highest.
classifier.show_most_informative_features()

#### Evaluate

def do_evaluation (pairs, pos_cls='pos', verbose=True):
    N = len(pairs)
    (ctr,correct, tp, tn, fp,fn) = (0,0,0,0,0,0)
    for (predicted, actual) in pairs:
        ctr += 1
        if predicted == actual:
            correct += 1
            if actual == pos_cls:
                tp += 1
            else:
                tn += 1
        else:
            if actual == pos_cls:
                fp += 1
            else:
                fn += 1
    (accuracy, precision, recall) = (float(correct)/N,float(tp)/(tp + fp),float(tp)/(tp + fn))
    if verbose:
        print_results(precision, recall, accuracy, pos_cls)
    return (accuracy, precision, recall)

def print_results (precision, recall, accuracy, pos_cls):
    banner =  'Evaluation with pos_cls = %s' % pos_cls
    print
    print banner
    print '=' * len(banner)
    print '%-10s %.1f' % ('Precision',precision*100)
    print '%-10s %.1f' % ('Recall',recall*100)
    print '%-10s %.1f' % ('Accuracy',accuracy*100)
    
    
pairs = [(classifier.classify(example), actual)
            for (example, actual) in test_set]

do_evaluation (pairs)
do_evaluation (pairs, pos_cls='neg')

#%% Other classifier : SVM ###################################################

# http://www.nltk.org/howto/classify.html
# Run example

from nltk.classify import SklearnClassifier
from sklearn.svm import SVC

t0 = time.time()
classif = SklearnClassifier(SVC(), sparse=False).train(train_set)
print(round(time.time()-t0,2))

classif.classify_many(test_set[0][0])

sizeTrain = [800]  # the first 100, the first 300 ,etc
testDoc = [800, 1000] # 800 to 999

classif.classify_many(test_set[0][0])

#%% SVM Class ################################################################

from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
    
class SVM:
    """ SVM
    
    data = dict (key = pos or neg), value = list of filenames
    mr = nltk.corpus.reader.plaintext.CategorizedPlaintextCorpusReader
    
    """

    def __init__(self, mr, data):
        self.mr = mr
        self.data = data
        self.classif = None
        self.predictions = None
        self.recall = None
        self.precision = None
        self.crosstab = None
        self.F = None
            
    def unigram_features (self,words):
        return dict((word, True) for word in words)
       
    
    def extract_features (self, corpus, file_ids, cls, feature_extractor=unigram_features):
                
        return [(feature_extractor(corpus.words(i)), cls) for i in file_ids]
        
    def trainModel(self, size):
        neg_training = self.extract_features(self.mr, self.data['neg'][:size], 'neg',
                                    feature_extractor=self.unigram_features)

    
        pos_training = self.extract_features(self.mr, self.data['pos'][:size],'pos',
                                        feature_extractor=self.unigram_features)
              
        
        train_set = pos_training + neg_training
        
        
        classif = SklearnClassifier(SVC(), sparse=False).train(train_set)
        
        self.classif = classif
        
        
    def predictDocuments(self, testDoc):
        """ Classify test documents
                
        args:
            test_doc (list): first value = starting file index, second = ending file index
        
        returns:
            pandas dataframe: rows = docID, columns = actual and predicted classes
        
        """
        
        neg_test = self.extract_features(self.mr, self.data['neg'][testDoc[0]:testDoc[1]], 'neg',
                                    feature_extractor=self.unigram_features)
                                    
        pos_test = self.extract_features(self.mr, self.data['pos'][testDoc[0]:testDoc[1]],'pos',
                                    feature_extractor=self.unigram_features)      
    
        test_set = pos_test + neg_test
        
        predictions = self.classif.classify_many([doc[0] for doc in test_set])
    
        results_dic = {}
        results_dic["actual"] = [t[1] for t in test_set]
        results_dic["predicted"] = predictions
        
        # dataframe
        df = pandas.DataFrame(results_dic, index = range(len(predictions)))  
        df.index.name = "docID"
        
        self.predictions = df
        
        
    def evaluate(self):
        """ Evaluate test by computing precision, recall, F
        
        """
        comb = [(x,y) for x in ["pos","neg"] for y in ["pos","neg"]]
        classification = ["tp", "fn", "fp", "tn"]
        
        classif_dict = {k: v for (k,v) in zip(classification, comb)}
        classif_counts = {k:  sum((self.predictions["actual"] == v[0])*(self.predictions["predicted"] == v[1]))
                                    for (k,v) in classif_dict.items() }
        
        crosstab = pandas.crosstab(self.predictions["actual"], self.predictions["predicted"])
        
        precision = classif_counts["tp"]/(classif_counts["tp"] + classif_counts["fp"])
        recall = classif_counts["tp"]/(classif_counts["tp"] + classif_counts["fn"])
        
        beta = 1
    
        F = (beta**2 + 1)*precision*recall/((beta**2)*precision + recall)
        
        self.recall = recall
        self.precision = precision
        self.crosstab = crosstab
        self.F = F
        
        
    def getPreditions(self):
        return self.predictions
        
    def getEvaluation(self):
        return {"recall": self.recall, "precision":self.precision, 
                "crosstab": self.crosstab, "F": self.F}
    
 
#%% Train and Test sample ####################################################
        
sizeTrain = [800]  # the first 100, the first 300 ,etc
testDoc = [800, 1000] # 800 to 999

MySVM = SVM(mr,data)        
MySVM.trainModel(sizeTrain[0])   

MySVM.predictDocuments(testDoc)
MySVM.evaluate()
MySVM.getEvaluation()
MySVM.getPreditions()


#%% Function to run various NB different training set #######################

def runSVM(sizeTrain, testDoc, mr, data):
    """ Run SVM
     
    args:
        data = dict (key = pos or neg), value = list of filenames
        mr = nltk.corpus.reader.plaintext.CategorizedPlaintextCorpusReader
        sizeTrain (list): list of size docuemnts train (e.g. 800 means first 8oo documents)
        test_doc (list): first value = starting file index, second = ending file index
        
    reutrns:
        dict: key = size, value = dict with key = F, crosstab, precision, recall, predictions
    
    """
    
    results = {k: None for k in sizeTrain}

    t0 = time.time()
    
    for size in sizeTrain:
        
        MySVM = SVM(mr,data)        
        MySVM.trainModel(size)   
        
        MySVM.predictDocuments(testDoc)
        MySVM.evaluate()
 
        results[size] = MySVM.getEvaluation()
        results[size]["predictions"] = MySVM.getPreditions()
        
        print("Size: {0}, Time so far: {1}".format(size, round(time.time()-t0,2)))
    
    return results
    

    
#%% Train and Test different sizes ###########################################
sizeTrain = [100, 300, 500, 600] # the first 100, the first 300 ,etc
testDoc = [600,800] # 600 to 799

SVMresults1 = runSVM(sizeTrain, testDoc, mr, data)
SVMresults_df1 = resultsToPandas(SVMresults1 )
SVMresults_df1

SVMresults_df1.to_csv("hw_2_6_performance1_SVM.txt", sep=',', encoding='utf-8')    
    

#%% Plotresults ###########################################

colors = {"precision": "b", "recall": "g", "F": "r"}

for column in SVMresults_df1.columns:
    plt.plot(SVMresults_df1.index,SVMresults_df1[column],'-{}o'.format(colors[column]), label=column)

# Place a legend above this legend, expanding itself to
# fully use the given bounding box.
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)

plt.xlabel('size of training set')
plt.ylabel('evluation metrics')
plt.xlim(0, 700)
plt.ylim(0.6,1)
plt.savefig('hw_2_6_performance1_SVM.png')
plt.show()


#%% Run on hold out test  ###########################################

sizeTrain = [800]  # the first 100, the first 300 ,etc
testDoc = [800,1000] # 600 to 799

SVMresults2 = runSVM(sizeTrain, testDoc, mr, data)
SVMresults_df2 = resultsToPandas(SVMresults2)
SVMresults_df2

SVMresults_df2.to_csv("hw_2_6_performance2_SVM.txt", sep=',', encoding='utf-8')    

SVMresults2[800]["predictions"].to_csv("hw_2_6_predictions2_SVM.txt", sep=',', encoding='utf-8')

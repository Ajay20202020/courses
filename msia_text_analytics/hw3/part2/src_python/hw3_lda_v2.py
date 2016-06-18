# -*- coding: utf-8 -*-
"""
MSiA - Text Analytics - Homework 3
Author: Luis Steven Lin
IDE: Spyder, Python 2

# References
# https://pypi.python.org/pypi/lda
# https://pythonhosted.org/lda/api.html
# http://chrisstrelioff.ws/sandbox/2014/11/13/getting_started_with_latent_dirichlet_allocation_in_python.html
"""

#%% setup  ##################################################################

import os
import json
#import re
#import sys
import pandas
import numpy as np
import lda
import lda.datasets
import time

path = 'C:\Users\llin\Downloads'
in_file_name = 'vectorFreq.json'
out_file_name_matrix = "td_matrix.csv"
out_file_name_matrix_sample = "td_matrix_sample.csv"
out_file_name_top10 = "top10_topics_words.csv"
out_file_name_topTopics = "top_doc_topics.csv"
os.chdir(path)
os.listdir('.') # see if file is in directory

#%% load data  ###############################################################

with open(in_file_name, 'r') as fp:
    data = json.load(fp)
    
# structure = {"Beijing_China}": {"hello": 25, "tomorrow": 2}}
    
# dictionary key: "Beijing_China", value = pandas.Series with value = count, index = terms
counts= {k: pandas.Series(v) for (k,v) in data.items()}

# create data frame index = terms, columns = document (city_country)
df = pandas.DataFrame(counts).fillna(0, inplace = False)
df.shape # (79652, 249)
df["Beijing_China"][df["Beijing_China"]>20]

# save 80 mb too big
df.to_csv(out_file_name_matrix, sep=",", header=True, index=True, encoding = 'utf-8')

# sample 
df[:1000].to_csv(out_file_name_matrix_sample, sep=",", header=True, index=True, encoding = 'utf-8')

#%%  lDA ####################################################################


tf_matrix = df.as_matrix()
tf_matrix.shape # (79652, 249)

# for lda, rows have to by documents, columns terms
tf_matrix = np.transpose(tf_matrix)
tf_matrix.shape
 
tf_matrix.dtype  # dtype('float64')          
tf_matrix = tf_matrix.astype(np.int64) # convert to integer
tf_matrix.dtype 

docs = df.columns
terms = df.index

####### Model #######
model = lda.LDA(n_topics=100, n_iter=1500, random_state=1)

# before training/inference:
SOME_FIXED_SEED = 0124
np.random.seed(SOME_FIXED_SEED)

t0 = time.time()
model.fit(tf_matrix)

t1 = time.time()
print(t1-t0) # 27 minutes

n_top_words = 10

####### topic-word probabilities ######
# a distribution over the N words in the vocabulary for each of the 20 
# topics. For each topic, the probabilities of the words should be normalized.
topic_word = model.topic_word_  # model.components_ also works

# topics x terms
print("type(topic_word): {}".format(type(topic_word)))
print("shape: {}".format(topic_word.shape))

# check normalized 
for n in range(5):
    sum_pr = sum(topic_word[n,:])
    print("topic: {} sum: {}".format(n, sum_pr))

####### top 10 words for each topic (by probability) ####### 

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(terms)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_words_decode = [t.encode('utf-8') for t in topic_words]
    print('Topic {}: {}'.format(i, ' '.join(topic_words_decode)))
    
####### document-topic probabilities ####### 
# distribution over the 20 topics for each of the 395 documents. 
# These should be normalized for each document
    
# documents x topics
doc_topic = model.doc_topic_
print("type(doc_topic): {}".format(type(doc_topic)))
print("shape: {}".format(doc_topic.shape))

# check normalized
for n in range(5):
    sum_pr = sum(doc_topic[n,:])
    print("document: {} sum: {}".format(n, sum_pr))

#######  sample the most probable topic ####### 
for i in range(len(docs)):
    print("{} - {} (top topic: {})".format(i,docs[i].encode('utf-8'), doc_topic[i].argmax()))

topTopics = [doc_topic[i].argmax() for i in range(len(docs))]  

dfTopTopics = pandas.DataFrame({"Document": docs, "Top_Topic": topTopics})  
dfTopTopics.index.name = "DocId"
dfTopTopics.to_csv(out_file_name_topTopics, sep=",", header=True, index=True, encoding = 'utf-8')
    
#%%  TOP WORDS ####################################################################

#Attributes

#components_	(array, shape = [n_topics, n_features]) Point estimate of the topic-word distributions (Phi in literature)
#topic_word_ :	Alias for components_
#nzw_	(array, shape = [n_topics, n_features]) Matrix of counts recording topic-word assignments in final iteration.
#ndz_	(array, shape = [n_samples, n_topics]) Matrix of counts recording document-topic assignments in final iteration.
#doc_topic_	(array, shape = [n_samples, n_features]) Point estimate of the document-topic distributions (Theta in literature)
#nz_	(array, shape = [n_topics]) Array of topic assignment counts in final iteration.

# get counts of topic assignments for each topic
model.nz_.shape # 100 topics

# check counts match
model.nzw_.sum(axis = 1).shape # sum across columns
model.nz_.shape  == model.nzw_.sum(axis = 1).shape 

topics = ["Topic " + str(i) for i in range(len(model.nz_))]

counts = pandas.Series(model.nz_, index = topics)

# get top 10 words for each topic (key = Topic i, value = string top 10 words)
dicTop10w = {}
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(terms)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_words_decode = [t.encode('utf-8') for t in topic_words]
    top10str = ' '.join(topic_words_decode)
    dicTop10w[topics[i]] =  top10str 
    
top10w = pandas.Series(dicTop10w)

# create data frame: index: topic, columns: counts, top 10 words
dfCounts = pandas.DataFrame( {"counts": counts, "top 10 words": top10w} )

# sort
dfCounts_sorted = dfCounts.sort(["counts"], ascending = [False])
dfCounts_sorted 

top10 = dfCounts_sorted[:10]
    
top10.to_csv(out_file_name_top10, sep=",", header=True, index=True, encoding = 'utf-8')

#%% Plotting: Takes a long time  ##############################################
import matplotlib.pyplot as plt

# use matplotlib style sheet
try:
    plt.style.use('ggplot')
except:
    # version of matplotlib might not be recent
    pass

# get index of top 5 topics
top5 = model.nz_.argsort()[-5:][::-1]
top5

t0 = time.time()

###### topic-word distribution ######
# The idea here is that each topic should have a distinct distribution of 
# words. In the stem plots below, the height of each stem reflects the
# probability of the word in the focus topic
f, ax= plt.subplots(5, 1, figsize=(8, 6), sharex=True)
for i, k in enumerate(top5):
    ax[i].stem(topic_word[k,:], linefmt='b-',
               markerfmt='bo', basefmt='w-')
    ax[i].set_xlim(-50,4350)
    ax[i].set_ylim(0, 0.08)
    ax[i].set_ylabel("Prob")
    ax[i].set_title("topic {}".format(k))

ax[4].set_xlabel("word")

plt.tight_layout()
plt.show()
plt.savefig('topic_word_distribution.png')
    
###### topic distribution for a few documents ######
# distributions give the probability of each of the 20 topics for every document
# many documents have more than one topic with high probability. As a result,
# choosing the topic with highest probability for each document can be subject to uncertainty
    
f, ax= plt.subplots(5, 1, figsize=(8, 6), sharex=True)
for i, k in enumerate([92, 139, 186, 197, 239]):
    ax[i].stem(doc_topic[k,:], linefmt='r-',
               markerfmt='ro', basefmt='w-')
    ax[i].set_xlim(-1, 21)
    ax[i].set_ylim(0, 1)
    ax[i].set_ylabel("Prob")
    ax[i].set_title("Document {}".format(k))

ax[4].set_xlabel("Topic")

plt.tight_layout()
plt.show()
plt.savefig('doc_topic_distribution.png')
    
t1 = time.time()
print(t1-t0) # 12 minutes
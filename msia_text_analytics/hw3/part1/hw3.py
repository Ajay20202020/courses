# -*- coding: utf-8 -*-
"""
MSiA - Text Analytics - Homework 3
Author: Luis Steven Lin

"""

# References
# http://www.nltk.org/book/ch03.html
# http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
# http://stackoverflow.com/questions/7152762/how-to-redirect-print-output-to-a-file-using-python
# http://stackoverflow.com/questions/2225564/get-a-filtered-list-of-files-in-a-directory
# http://matthewrocklin.com/blog/work/2014/05/01/Fast-Data-Structures/
# http://pandas.pydata.org/pandas-docs/stable/dsintro.html
# http://pandas.pydata.org/pandas-docs/version/0.15.2/indexing.html
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html
# http://stackoverflow.com/questions/25736861/python-pandas-finding-cosine-similarity-of-two-columns
# http://stackoverflow.com/questions/28883303/calculating-similarity-between-rows-of-pandas-dataframe
# http://stackoverflow.com/questions/17627219/whats-the-fastest-way-in-python-to-calculate-cosine-similarity-given-sparse-mat
# http://stackoverflow.com/questions/19477264/how-to-round-numpy-values-in-savetxt
# http://stackoverflow.com/questions/5469286/how-to-get-the-index-of-a-maximum-element-in-a-numpy-array-along-one-axis


#%% Setup ###################################################################
import os
import re
from nltk.corpus import stopwords
from stanford_corenlp_pywrapper import CoreNLP
from __future__ import division # division results in float if both integers
import pandas
from scipy.spatial.distance import cosine
import math
from sklearn.metrics import pairwise_distances
import sys
import numpy as np

path = '/Users/Steven/Documents/Academics/3_Graduate School/2014-2015_NU/MSIA_490_Text_Mining/hw/hw3'
in_file_name = 'classbios.txt'
os.chdir(path)
os.listdir('.') # see if file is in directory

# create directory for output
out_file_bios_folder = "classbios_by_person"
if not os.path.exists(out_file_bios_folder):
    os.makedirs(out_file_bios_folder)
    
# create directory for output
out_file_bios_folder_normalized = "classbios_by_person_normalized"
if not os.path.exists(out_file_bios_folder_normalized):
    os.makedirs(out_file_bios_folder_normalized)
    
    
#%% Functions ###############################################################
    
def getCounts(doc):
    """ Gets the frequencies for elements in document
    
    args:
        doc (list) = list of tokens in document
    returns:
        pandas.core.series.Series: index = element, value = count
    """
    series = pandas.Series(doc)
    c = series.value_counts()
    return c
    
def getTermDocument(doc_dic_normalized):
    """ Term Document Dataframe, with column = document name, row = terms
    
    args:
        doc_dic_normalized (dic) = key: document name, value: list of terms
    returns:
        pandas.DataFrame with index = terms, column name = document name
    """    
       
    # key = document_name, value = pandas series counts
    counts = {name: getCounts(doc) for (name,doc) in doc_dic_normalized.items()}
    
    # Vocabulary = distinct words from all documents
    vocabulary = set([v for k in counts for v in counts[k].index])
    len(vocabulary) # 1769
    
    # for each class, fill missing values for words of that class that are not in the vocabulary
    # do this by creating a pandas series from the current count series using vocabulary
    # as index. The resulting series will include all the words in the vocabulary as index
    # with value equal to the count in the orignal series, except for words that are not
    # in the original series which will get a NaN value, which in turns gets filled with zero
    counts_fill = {k: pandas.Series(v, index = vocabulary).fillna(0, inplace=False) for k,v in counts.items()}
    
    # actually previous step not needed, can include vectors with different elements
    # in data frame, and any missing elements for other documents will have NaN values (fill with zeros)
    
    # column name = document name, row names = token
    # df = pandas.DataFrame(counts , index = vocabular).fillna(0, inplace = False) # no need to provide index
    df = pandas.DataFrame(counts).fillna(0, inplace = False)
    
    return df


def convertBoolean(term_document):
    """ Term Document Binarized (counts > 1 have value 1, else 0)
    
    args:
        term_document (pandas.DataFrame): with index = terms, column name = document name
    returns:
        pandas.DataFrame with index = terms, column name = document name
    """    
    
    term_document_boolean = term_document.copy(deep = True)
    term_document_boolean[term_document_boolean>1] = 1
    
    return term_document_boolean
 
def findMaxPairs(df):
    """ Returns the most similar document for each document
    
    args:
        df (pandas.Dataframe): row, col = documents, value = boolean similarity
    
    returns:
        pandas.Dataframe: most similar document for each document
    
    """
    df2 = df.copy(deep=True)
    np.fill_diagonal(df2.values, -1)
    return df2.idxmax()
    
def findMinPairs(df):
    
    """ Returns the least similar document for each document
    
    args:
        df (pandas.Dataframe): row, col = documents, value = boolean similarity
    
    returns:
        pandas.Dataframe: least similar document for each document
    
    """
    return df.idxmin() 
    
 
def rankSimilarity(df, top = True, rank = 3):
    
    """ Returns the most similar documents or least similar documents
    
    args:
        df (pandas.Dataframe): row, col = documents, value = boolean similarity
        top (boolean): True: most, False: least (default = True)
        rank (int): number of top or bottom (default = 3)
    
    returns:
        pandas.Dataframe: row =rank, columns = indices, names, value
    
    """
    df2 = df.copy(deep = True)
    df_np = df2.as_matrix()
    
    if top:
        np.fill_diagonal(df_np, -1)
    
    results_dic = {"indices": [], "names": [], "value": [] }
    
    for n in range(rank):
        
        if top:
            indices = np.unravel_index(df_np.argmax(), df_np.shape) # returns indices of first max found
            # np.where(df_np == df_np.max()) # will return all indices of maxs
        else:
            indices = np.unravel_index(df_np.argmin(), df_np.shape) # returns indices of first min found
            # np.where(df_np == df_np.min()) # will return all indices of mins
            
        results_dic["indices"].append(indices)
        results_dic["names"].append((df.index[indices[0]], df.index[indices[1]]))
        results_dic["value"].append(df.iloc[indices])
        
        if top:
            df_np[indices[0],indices[1]] = -1 # set to -1 to find the next max
            df_np[indices[1],indices[0]] = -1 # because symmetric
        
        else:
            df_np[indices[0],indices[1]] = 1 # set to 1 to find the next min
            df_np[indices[1],indices[0]] = 1 # because symmetric
        
    df_result = pandas.DataFrame(results_dic, index = range(1,rank+1))  
    df_result.index.name = "rank"   
    
    return df_result
        
    
#%% Data Loading ############################################################
    
infile = open(in_file_name, 'r')  # Open the file for reading.
data = infile.read() # load all data at once
# text = nltk.data.load(os.path.join(path, in_file_name)) # read data at once in unicode
text = data.decode("ascii" , "ignore").encode('utf-8')# decode the input, ignore non-ascii, encode utf-8

# get the names
file_names = re.findall(r'==>.+<==',text ) 

# 1) ==> mays--jacob-late_25744_1488166_hw0.txt <==
# 2) mays--jacob-late (beause search finds first match)
#    special case : 'zhao--lily--qian-' ends in - so it is going to be captured
#    if want to match exact pattern:  \b[a-z]+-+[a-z]+(?:-+[a-z]+)?
# 3) replace -- by _
# 4) replace -late with empty space
# Note: combined multiple last names: jung-hee -> junghee, avalos-mar -> avalosmar

doc_names = [re.search(r"\b[^_]*", name)
                .group(0)
                .replace("--","_")
                .replace("-","")
                .replace("late","")
                for name in file_names ]
                    
documents = re.split(r'==>.+<==',text)
documents = documents[1:] # remove first empty line
len(documents) # 30

# key: name, value = index number
map_name_index = {name: index for (name,index) in zip(doc_names, range(len(doc_names)))}

# create dictionary key: doc name, value =docment
doc_dic = {name: doc for (name,doc) in zip(doc_names,documents)}
    
# save documents
for name in doc_dic:
    f = open(os.path.join(out_file_bios_folder,name + ".txt"), "w")
    f.write(doc_dic[name])
    f.close()

#%% Text Processing ##########################################################

proc = CoreNLP("pos", corenlp_jars=["/Users/Steven/Documents/corenlp/stanford-corenlp-full-2015-04-20/*"])

# You can also specify the annotators directly. For example, say we want to 
# parse but don't want lemmas. This can be done with the configdict option:
p = CoreNLP(configdict={'annotators':'tokenize, ssplit, pos, parse'}, 
            output_types=['pos','parse'],
            corenlp_jars=["/Users/Steven/Documents/corenlp/stanford-corenlp-full-2015-04-20/*"])

doc_dic_normalized = {} # key: document name, value = list of lemmas 
# note: remove stopwords, punctuation, numbers, websites, -lrb-, -rrb-

# this pattern is only going to match two cases of words: data, data-driven
# so ignores punctuation, numbers, parenthesis -rrb-, -lrb-, special characters
# ignore so use match instead of search
# match: Determine if the RE matches at the beginning of the string.
# ^ = beginning of string, $ = end of string so https://www.coursera.org is ignored
pattern = re.compile(r'^[A-Za-z]+[-]?[A-Za-z]+$')

# stopwords list , add "I" since Stanford NLP does not lowercase I but stopwords (it lowercases she)
# from nltk includes "i"
stop = set(stopwords.words('english'))
stop.add("I")

for doc_name in doc_names:
    
    # Now it's ready to parse documents. You give it a string and it returns JSON-safe data structures
    # dictionary key = 'sentences', value = list of sentences
    # each sentence dictionary with key='lemmas', 'tokens', etc
    # key = 'lemmas', value = list of lemmas 
    parsed = proc.parse_doc(doc_dic[doc_name])["sentences"]
    sentences = [sentence["lemmas"] for sentence in parsed]
    #flatten nested list so each element is a token
    doc_dic_normalized[doc_name] = [lemma for sentence in sentences for lemma in sentence 
        if pattern.match(lemma) and lemma not in stop]
    
# count number of tokens: 5256
len([v for ls in doc_dic_normalized.values() for v in ls])
    
# save documents
for name in doc_dic:
    f = open(os.path.join(out_file_bios_folder_normalized , name + ".txt"), "w")
    f.write(" ".join(doc_dic_normalized[name]))
    f.close()
    

#%% Boolean Similarity ##########################################################

term_document = getTermDocument(doc_dic_normalized)
term_document.index 
term_document.columns
term_document.shape #1769x30

term_document_boolean = convertBoolean(term_document)

# Note that spatial.distance.cosine computes the distance, and not the similarity.
# So, you must subtract the value from 1 to get the similarity.
# Note also need to transpose because pairwise distances does it row by row
term_document_boolean_tp = term_document_boolean.transpose()
similarity_boolean = 1-pairwise_distances(term_document_boolean_tp, metric="cosine")
np.shape(similarity_boolean) # 30 by 30

df_similarity_boolean = pandas.DataFrame(similarity_boolean , index=term_document_boolean.columns, columns=term_document_boolean.columns)
df_similarity_boolean["lin_luis"]["lin_luis"] # 1
df_similarity_boolean["lin_luis"]["li_xiang"] # 0.1856

# save results
np.savetxt("boolean.txt", similarity_boolean , fmt='%.3f', delimiter = " ")
df_similarity_boolean.to_csv("boolean_w_headers.csv", sep=",", header=True, index=True, float_format= '%.3f')

max_pairs = findMaxPairs(df_similarity_boolean)
min_pairs = findMinPairs(df_similarity_boolean)
max_pairs
min_pairs

max_pairs["lin_luis"] # 'bhargava_anuj'
min_pairs["lin_luis"] # 'leboeuf_mark'

map_name_index["lin_luis"] # 15
map_name_index[max_pairs["lin_luis"]] # 2
map_name_index[min_pairs["lin_luis"]] # 13
        
topSimilar = rankSimilarity(df_similarity_boolean, top = True, rank =3) 

bottomSimilar = rankSimilarity(df_similarity_boolean, top = False, rank =3)

# save results
topSimilar.to_csv("boolean_top.csv", sep=",", header=True, index=True)
bottomSimilar.to_csv("boolean_bottom.csv", sep=",", header=True, index=True)

# save output to output.txt
orig_stdout = sys.stdout
f = open('boolean_output.txt', 'w')
sys.stdout = f

print("\n********************************\n")
print("\n*Top****************************\n")

print(topSimilar)

print("\n*Bottom*************************\n")

print(bottomSimilar)

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))
                                       
f.close()
sys.stdout =  orig_stdout       


print("\n********************************\n")
print("\n*Top****************************\n")

print(topSimilar)

print("\n*Bottom*************************\n")

print(bottomSimilar)

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))

#%% Manual calculations, less efficient #######################################
# check cosine formula
"""
x = term_document_boolean["lin_luis"]
y = term_document_boolean["li_xiang"]
sum(pandas.Series.multiply(x,y)) #20
np.dot(x, y) #20
sum(i*j for (i,j) in zip(x,y)) #20

math.sqrt(sum(i**2 for i in x)) # 12.6
math.sqrt(sum(i**2 for i in y)) # 8.54
sum(i*j for (i,j) in zip(x,y))/(math.sqrt(sum(i**2 for i in x))*math.sqrt(sum(i**2 for i in y)))

np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))
1 - cosine(x,y)
"""

# do manually
"""
m = term_document_boolean.shape[1]
mat = np.zeros((m, m))
np_term_document_boolean = term_document_boolean.as_matrix()
np.shape(np_term_document_boolean)

for i in range(m):
    for j in range(i,m):
        if i != j:
            mat[i][j] = 1 - cosine(np_term_document_boolean[:,i], np_term_document_boolean[:,j])
            mat[j][i] = mat[i][j]
        else:
            mat[i][j] = 1
            
np.sum(mat-similarity_boolean) # basically zero
"""

#%% tf_idf ##################################################################

# term's raw frequency in this document
tf = 1+np.log10(term_document) # 1+log(tf) if tf > 0
tf = tf.replace([-np.inf],0)   # 0 otherwise

# the number of documents the term occurs in
# index = term, value = count documents
df = term_document_boolean.sum(axis=1)

# check
sum(term_document_boolean.loc['Analytics',:]) == df["Analytics"]

# N the total number of documents
N = term_document.shape[1]

idf = np.log10(N/df)
tf_idf = tf.multiply(idf,axis = 'index')

# check
sum(tf_idf["lin_luis"]) == sum(tf["lin_luis"]*idf)
tf_idf.loc["yelp","lin_luis"] == tf.loc["yelp","lin_luis"]*idf["yelp"]

# Note that spatial.distance.cosine computes the distance, and not the similarity.
# So, you must subtract the value from 1 to get the similarity.
# Note also need to transpose because pairwise distances does it row by row
term_document_tfidf_tp = tf_idf.transpose()
similarity_tfidf = 1-pairwise_distances(term_document_tfidf_tp, metric="cosine")
np.shape(similarity_tfidf) # 30 by 30

df_similarity_tfidf = pandas.DataFrame(similarity_tfidf , index=tf_idf.columns, columns=tf_idf.columns)
df_similarity_tfidf["lin_luis"]["lin_luis"] # 1
df_similarity_tfidf["lin_luis"]["li_xiang"] # 0.048035582616980377

# save results
np.savetxt("tf_idf.txt", similarity_tfidf , fmt='%.3f', delimiter = " ")
df_similarity_tfidf.to_csv("tf_idf_w_headers.csv", sep=",", header=True, index=True, float_format= '%.3f')

max_pairs = findMaxPairs(df_similarity_tfidf)
min_pairs = findMinPairs(df_similarity_tfidf)
max_pairs
min_pairs

max_pairs["lin_luis"] # 'bhargava_anuj'
min_pairs["lin_luis"] # 'fassois_demetrios'

map_name_index["lin_luis"] # 15
map_name_index[max_pairs["lin_luis"]] # 2
map_name_index[min_pairs["lin_luis"]] # 6
        
topSimilar = rankSimilarity(df_similarity_tfidf, top = True, rank =3) 

bottomSimilar = rankSimilarity(df_similarity_tfidf, top = False, rank =3)

# save results
topSimilar.to_csv("tf_idf_top.csv", sep=",", header=True, index=True)
bottomSimilar.to_csv("tf_idf_bottom.csv", sep=",", header=True, index=True)

# save output to output.txt
orig_stdout = sys.stdout
f = open('tf_idf_output.txt', 'w')
sys.stdout = f

print("\n********************************\n")
print("\n*Top****************************\n")

print(topSimilar)

print("\n*Bottom*************************\n")

print(bottomSimilar)

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))
                                       
f.close()
sys.stdout =  orig_stdout       


print("\n********************************\n")
print("\n*Top****************************\n")

print(topSimilar)

print("\n*Bottom*************************\n")

print(bottomSimilar)

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))

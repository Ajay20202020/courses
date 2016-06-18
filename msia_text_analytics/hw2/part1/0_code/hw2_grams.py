# -*- coding: utf-8 -*-
"""
MSiA - Text Analytics - Homework 2
Author: Luis Steven Lin
IDE: Spyder, Python 2

Tokenization, Lemmatization, Unigram, Bigram

https://github.com/brendano/stanford_corenlp_pywrapper

# References
# http://www.nltk.org/book/ch03.html
# http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
# http://stackoverflow.com/questions/7152762/how-to-redirect-print-output-to-a-file-using-python

# Note: Stanford NLP does not have stop word removal. So have to remove in python code using
# stopwords from nltk

# Note: Stanford NLP converts to lower cases regulars words (e.g does not 
# convert lower case I, China, etc). So need to add "I" to stopwords

# Note: Stanford NLP: parentheis are -rrb- and -lrb-

# Note: also ignore non-ascii characters, remove leading and trailing whitespace

# Note: consider punctuation and stopwords for bigrams construction, 
# but remove bigrams with punctuation or stopwords. It is not correct to remove 
# them before constructing the bigrams (e.g. I liked the movie--> liked movide 
# --> (liked,movie) is not a correct bigram )

"""

#%% setup  ##################################################################

import os
import json
from nltk.corpus import stopwords
from stanford_corenlp_pywrapper import CoreNLP
import re
import sys

path = '/Users/Steven/Documents/Academics/3_Graduate School/2014-2015_NU/MSIA_490_Text_Mining/hw/hw2'
in_file_name = 'classbios.txt'
split = in_file_name.split(".")
out_file_name_lines = split[0] + "_lines." + split[1]
out_file_name_normalized_line = split[0] + "_normalized_line ." + split[1]
out_file_name_normalized_sentence = split[0] + "_normalized_sentence ." + split[1]
out_file_name_normalized_tokens = split[0] + "_normalized_tokens ." + split[1]

k = 20 # top k unigrams and bigrams

out_file_name_unigrams = "{0}_top{1}{2}.json".format(split[0],k,"Unigrams")
out_file_name_bigrams = "{0}_top{1}{2}.json".format(split[0],k,"Bigrams")

os.chdir(path)
os.listdir('.') # see if file is in directory

proc = CoreNLP("pos", corenlp_jars=["/Users/Steven/Documents/corenlp/stanford-corenlp-full-2015-04-20/*"])

# You can also specify the annotators directly. For example, say we want to 
# parse but don't want lemmas. This can be done with the configdict option:
p = CoreNLP(configdict={'annotators':'tokenize, ssplit, pos, parse'}, 
            output_types=['pos','parse'],
            corenlp_jars=["/Users/Steven/Documents/corenlp/stanford-corenlp-full-2015-04-20/*"])
            
#%% Functions  ##############################################################


def getFrequency(ls, ignore = set(), pattern = re.compile(r'.') ):
    """Gets the frequency of elements in list, ignoring elements in ignore set
    and matching the pattern
    
    Args:
        ls(list): list of items
        ignore(set): elements to ignore (default: empty set)
        pattern (re.compile): pattern to match (default: everything)
    
    Returns:
        dict: dictionary key: item, value: frequency (count)
    
    """
    freq = {}
    
    for element in ls:
        
        if element not in ignore and pattern.match(element):
            if element not in freq:
                freq[element] = 1
            else:
                freq[element] += 1

    return freq
    

def getFrequencyBigram(ls, ignore = set(), pattern = re.compile(r'.')):
    """Gets the frequency of pairs of consecutive elements in the list, 
       ignoring pairs with at an element in ignore set and matching the pattern
    
    Args:
        ls(list): list of items
        ignore(set): elements to ignore (default: empty set)
        pattern (re.compile): pattern to match (default: everything)
    
    Returns:
        dict: dictionary key: item, value: frequency (count)
    
    """
    freq = {}
    
    for i in range(len(ls)-1):
        
        # could be improved, don't need to recheck for i+1
        if (ls[i] not in ignore and ls[i+1] not in ignore
            and pattern.match(ls[i]) and pattern.match(ls[i+1])):

            key = (ls[i],ls[i+1])
            
            if key not in freq:
                freq[key] = 1
            else:
                freq[key] += 1

    return freq


def findTop(freq, k):
    """ Find top k keys and its frequencies from freq dictionary 
    Note that if there are ties for the values for k, it includes all those values as welll
    (So the function might return more than k elements)
    
    Args:
        freq (dict): frequency dictionary key: item, value: frequency
        k (int): top k
        
    Returns
        list: top k items with its frequencies
    
    """
    
    freq_list = list(freq.items())
    max_list = []
    
    i = 0
    done = False # true when no more elements to add because the value is less than that of k
    
    # iterate until k number of max values are removed
    while i < len(freq_list) and not done:
           
        max_value = freq_list[0][1]
        max_index = 0
        
        for j in range(1,len(freq_list)):
            
            # update max
            if freq_list[j][1] > max_value:
                max_index = j
                max_value = freq_list[j][1]
        
        # add until k elements are added        
        if i < k:        
            # remove max tuple and add to max        
            max_list.append(freq_list.pop(max_index))
        
        # keep adding until the value of elements following k is less than that value for k
        # (add while the following elements have the same value as k)
        elif max_value == max_list[i-1][1]:
            max_list.append(freq_list.pop(max_index))
            
        else:
            done = True
                
        i+=1
              
    
    # sort by count
    max_list.sort(key = lambda x : -x[1])
    
    return max_list
    
    # return only keys
    #return list(zip(*max_list))[0]
            
#%% Load data  ###############################################################

# remove lines starting with ==>
# ignore non-ascii
# http://stackoverflow.com/questions/3667875/removing-non-ascii-characters-from-any-given-stringtype-in-python

infile = open(in_file_name, 'r')  # Open the file for reading.
# data = infile.read()  # load all data at once
# text = nltk.data.load(os.path.join(path, in_file_name)) # read data at once in unicode
# text_split = text.split('\n')

# .encode converts Unicode objects into strings, and .decode converts strings into Unicode.
text = [line.decode("ascii" , "ignore").encode("ascii") for line in infile if line[0:3]!='==>']

infile.close()

# save processed text line by line in a file
f = open(out_file_name_lines, "w")
for line in text:
    f.write(line) # no need to add "/n" because already in string
f.close()

#%% Tokenize  ################################################################

# sentence segmentation of each line (each element is a list of sentences)
# Seprating by sentene is ok because bigrams like (".", "I") will be ignored
tokenized_lines = []     # each element is a list with tokens of a line
tokenized_sentences = [] # each element is a list with tokens of sentence

for line in text:
    # Now it's ready to parse documents. You give it a string and it returns JSON-safe data structures
    # dictionary key = 'sentences', value = list of sentences
    # each sentence dictionary with key='lemmas', 'tokens', etc
    # key = 'lemmas', value = list of lemmas 
    parsed = proc.parse_doc(line)["sentences"]
    sentences = [sentence["lemmas"] for sentence in parsed]
    
    # flatten nested list so each element is a token
    line_tokenized = [sentence for sublist in sentences for sentence in sublist]
    # add to list where each element is a tokenized line
    if line_tokenized != []:
        tokenized_lines.append(line_tokenized)
    
    # add to list where each element is a tokenized sentence
    for sentence in sentences:
        tokenized_sentences.append(sentence)

# save to file
len(tokenized_lines) # 182 lines
f = open(out_file_name_normalized_line, "w")
for line in tokenized_lines:
    f.write(" ".join(line) + "\n")
f.close() 

# save to file
len(tokenized_sentences) # 500 sentences
f = open(out_file_name_normalized_sentence, "w")
for sentence in tokenized_sentences:
    f.write(" ".join(sentence) + "\n")
f.close()    

# convert to a list of tokens
tokens = [token for line in tokenized_lines for token in line]
len(tokens) # 10379 tokens
f = open(out_file_name_normalized_tokens, "w")
for token in tokens:
    f.write(token  + "\n")
f.close()    


#%% Saved and Write Json  ####################################################

"""
with open('data.json', 'w') as fp:
    json.dump(x, fp)

with open('data.json', 'r') as fp:
    data = json.load(fp)

"""
#%% Data processing  #########################################################

# stopwords list , add "I" since Stanford NLP does not lowercase I but stopwords
# from nltk includes "i"
stop = set(stopwords.words('english'))
stop.add("I")
    
# remove leading and trailing whitespace just in case and non-ascii characters
#text_strip = [element.strip().encode('ascii',errors='ignore') for element in text_split]
tokens_strip = [element.strip() for element in tokens]

# do NOT do here at this stage, but rather during frequency count
# remove stop words
# tokens_nostop = [element for element in tokens_strip if element not in stop]

# regex: only keep words composed of alphanumeric characters or alphanumeric 
# words joined by "-" (e.g. keep data-driven)
# ignore parenthesis -rrb-, -lrb- so use match instead of search
# match: Determine if the RE matches at the beginning of the string.

pattern = re.compile(r'[A-Za-z0-9]+[- ][A-z0-9]+|[A-Za-z0-9]+')

# do NOT do here at this stage, but rather during frequency count
#tokens_clean = [element for element in tokens_nostop if pattern.match(element)]


#%% Frequencies Unigrams

# dictionary key = words, value = count
freqUnigram = getFrequency(tokens_strip, ignore = stop, pattern = pattern) 
topKUnigram = findTop(freqUnigram, k) # list of top 20 words

with open(out_file_name_unigrams, 'w') as fp:
    json.dump(topKUnigram, fp)


#%% Frequencies Bigrams

freqBigram = getFrequencyBigram(tokens_strip,ignore=stop, pattern = pattern)
topKBigram = findTop(freqBigram, k)

with open(out_file_name_bigrams, 'w') as fp:
    json.dump(topKBigram, fp)

#%% Print and Save results

# save output to output.txt
orig_stdout = sys.stdout
f = open('output.txt', 'w')
sys.stdout = f

print("\n********************************\n")
print("- Number of lines: {0}\n"
      "- Number of sentences: {1}\n"
      "- Number of tokens: {2}".format(len(tokenized_lines),
                                       len(tokenized_sentences),
                                       len(tokens)))
                                       
print("\n********************************\n")
print("Number of unigrams (after removing unigrams with stop words and punctuation): {}\n".format(len(freqUnigram)))
print("Top {0} Unigrams:\n".format(k))

for tup in topKUnigram:
    print("{0:>15} : {1}".format(tup[0],str(tup[1])))

print("\n*Note: if there are ties for the values for k, it includes all those values as well. "
      "(So the function might return more than k elements)")

print("\n********************************\n")
print("Number of Bigrams (after removing bigrams with stop words and punctuation): {}\n".format(len(freqBigram)))

print("Top {0} Bigrams:\n".format(k))

for tup in topKBigram:
    bigram = "{0}, {1}".format(*tup[0])
    print("{0:>25} : {1}".format(bigram,str(tup[1])))

print("\n*Note: if there are ties for the values for k, it includes all those values as well. "
      "(So the function might return more than k elements)")

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))
                                       
f.close()
sys.stdout =  orig_stdout                    

#%% Display Results

print("\n********************************\n")
print("- Number of lines: {0}\n"
      "- Number of sentences: {1}\n"
      "- Number of tokens: {2}".format(len(tokenized_lines),
                                       len(tokenized_sentences),
                                       len(tokens)))
                                       
print("\n********************************\n")
print("Number of unigrams (after removing unigrams with stop words and punctuation): {}\n".format(len(freqUnigram)))
print("Top {0} Unigrams:\n".format(k))

for tup in topKUnigram:
    print("{0:>15} : {1}".format(tup[0],str(tup[1])))

print("\n*Note: if there are ties for the values for k, it includes all those values as well. "
      "(So the function might return more than k elements)")

print("\n********************************\n")
print("Number of Bigrams (after removing bigrams with stop words and punctuation): {}\n".format(len(freqBigram)))

print("Top {0} Bigrams:\n".format(k))

for tup in topKBigram:
    bigram = "{0}, {1}".format(*tup[0])
    print("{0:>25} : {1}".format(bigram,str(tup[1])))

print("\n*Note: if there are ties for the values for k, it includes all those values as well. "
      "(So the function might return more than k elements)")

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))
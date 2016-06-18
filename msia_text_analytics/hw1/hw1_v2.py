# -*- coding: utf-8 -*-
"""
MSiA - Text Analytics - Homework 1
Author: Luis Steven Lin
IDE: Spyder, Python 3

"""

# References
# http://www.nltk.org/book/ch03.html
# http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python

import os
import nltk
import re

#%% Setup

path = '/Users/Steven/Documents/Academics/3_Graduate School/2014-2015_NU/MSIA_490_Text_Mining/hw/hw1'
in_file_name = 'classbios.txt'
#in_file_name = 'test_reg_ex.txt'
out_file_name = 'classbios_timepoints.txt'

os.chdir(path)
os.listdir('.') # see if file is in directory


#%% Data Processing

# read the file as string
f = open(in_file_name, encoding = 'latin2')
text = f.read()

# split by new line (takes care of headers before bios)
text_split = text.split('\n')

# sentence segmentation of each line (each element is a list of sentences)
sents_nested = [nltk.sent_tokenize(line) for line in text_split if line != '']
# sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

# flatten nested list so each element is a sentence
sents = [sentence for sublist in sents_nested for sentence in sublist]

# check number of senteces
len(sents) # 543

#%% Regular Expressions

months_list = ["[Jj]an(?:uary)?","[Ff]eb(?:ruary)?", "[Mm]ar(?:ch)?",
          "[Aa]pr(?:il)?", "[Mm]ay","[Jj]un(?:e)?", "[Jj]ul(?:y)?",
          "[Aa]ug(?:ust)?","[Ss]ep(?:tember)?", "[Oo]ct(?:ober)?",
          "[Nn]ov(?:ember)?","[Dd]ec(?:ember)?" ]
          
seasons_list = ["[Ss]ummer","[Ff]all","[Ww]inter","[Ss]pring","[Qq]uarter]"]

adjectives_list = ["next", "past", "last", "this"]

months = "|".join(months_list)
seasons =  "|".join(seasons_list)
adjectives = "|".join(adjectives_list)

# Example yyyy/dd/mm
pattern1 = re.compile(r"\b[0-9]{4}[-/.][0-9]?[0-9][-/.][0-9]?[0-9]")

# Example mm/dd/yyyy
pattern2 = re.compile(r"\b[0-3]?[0-9][/.-][0-3]?[0-9][/.-](?:[0-9]{2})?[0-9]{2}")

# Example 2015Jan02
# {{ }} to escape replacement
pattern3 = re.compile(r"\b[0-9]{{4}}(?:[-/. ]?|. )(?:{months})[-/. ]?[0-9]?[0-9]".format(months=months))
 
# January 30th, 2015
 # {{ }} to escape replacement
pattern4 = re.compile(r"\b(?:{months})[-/. ]?[ ]?[0-9]?[0-9] ?(?:st|nd|rd|th)?,? ?[0-9]{{4}}".format(months=months))

# Example: 31st October, 2015
 # {{ }} to escape replacement
pattern5 = re.compile(r"\b[0-9]?[0-9][ ]?(?:(?:th|st|nd|rd) (?:of)?)?"
                       "(?:[-/. ]?|. )(?:{months})"
                       "(?:[-/. ]| AD | BC )?(?:[0-9]{{2}})?[0-9]{{2}}".format(months=months))


# Example: December 2015, fall 2005, in 2010
 # {{ }} to escape replacement
pattern6 = re.compile(r"\b(?:{months}|"
                       "{seasons}|[Ii]n )[-/. ]?[0-9]{{4}}".format(months=months,seasons=seasons))
                      
# Example: this fall, next spring, last summer, the past summer, 2014-2015
 # {{ }} to escape replacement
pattern7 = re.compile(r"\b(?:{adjectives}) (?:{seasons})|\b[0-9]{{4}}-[0-9]{{4}}".
                        format(seasons=seasons, adjectives = adjectives))
                       
pattern_list = [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7]

#f = open(out_file_name, "w")
f = open(out_file_name, "w", encoding = 'utf-8')


i=0
for sentence in sents:
    
    # search regular expression
    # Note: a more efficient way would be to create a big regular expression
    # with or statetments for each pattern. But here this was done in this
    # way in case extracting particular pattern match is required
    match_list = [x.search(sentence) for x in pattern_list]

    # if any matches print line (if any TRUE in TRUE/FALSE list)
    if any([obj != None for obj in match_list]):
        print(sentence)
        f.write(sentence + '\n')
        #f.write(sentence.encode('ascii', 'ignore').decode('ascii') + "\n")
        i+=1
f.close()

print("\n\nNumber of matches: : {}".format(i))
print("Sentences with a match are saved in: {}".format(os.path.join(path, out_file_name)))

#%%

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 18:51:35 2015

@author: Steven

# References
http://www.crummy.com/software/BeautifulSoup/bs3/documentation.html
http://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text
"""

from bs4 import BeautifulSoup
import re
from urllib2 import urlopen #python 2
import time
from __future__ import division # division results in float if both integers
import os
import json

path = '/Users/Steven/Documents/Academics/3_Graduate School/2014-2015_NU/MSIA_490_Text_Mining/hw/hw3'
#in_file_name = 'classbios.txt'
out_file_name = 'webscraped_data.json'
os.chdir(path)
os.listdir('.') # see if file is in directory

#%% Functions ################################################################

def findIndex(listTags, pattern = re.compile(r'(declared|official|legislative|constitutional)')):
    ''' Find the index of the first match
    
    Args:
        listTags: list of beautiful soup tags
        pattern(regular expression): to match, default provided
        
    Returns:
        (int) first index
        
    '''
    index = [i for i, description in enumerate(listTags)
             if pattern.search(description.get_text())]
    return index[0]
    
def getLink(url):
    """ Gets the text from url
    
    Args:
        url (string)
    Returns:
        bs4.element.ResultSet
    
    """
    html = urlopen("http://en.wikipedia.org"+url)
    bsObj = BeautifulSoup(html)
    return bsObj.findAll(text=True)
    
def visible(element):
    ''' Tests if element is visible or not. Ignore non-ascii
    
    Args:
        element (bs4.element.Doctype)
        
    Returns:
        boolean: True if visible, False if not
    
    Reference: http://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text
    '''
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match(r'<!--.*-->', str(element.encode('ascii','ignore'))):
        return False
    return True
    
    
   
def processLink(url):
    ''' Keeps visible elements in url, ignores non-ascii, 
    strip space and ignores empty elements and elements with with size 1
    that are non-alphanumeric
        
    Args:
        url (string)
    Returns:
        string: processed text
    
    '''
    pattern = re.compile(r'[A-Za-z0-9]')
    results = getLink(url)
    results = filter(visible, results)
    results = [element.encode("ascii" , "ignore").decode('utf-8').strip() for element in results]
    results = [element for element in results if len(element) > 1 or pattern.match(element)]
    return " ".join(results)
    
    
def processLinkByParagraph(url):
    """ Gets the html of the link and processes by paragraph
    
    Args:
        url (string)
        
    Returns:
        string: processed text
    
    """
    
    html = urlopen("http://en.wikipedia.org"+url)
    bsObj = BeautifulSoup(html)
    paragraphs = bsObj.findAll('p')
    paragraphs = [p.get_text() for p in paragraphs]
    return " ".join(paragraphs)
    
          
#%% Webscrape ################################################################

t0 = time.time()
           
wiki_url = 'https://en.wikipedia.org/wiki/List_of_national_capitals_in_alphabetical_order'
html = urlopen(wiki_url)
bsObj = BeautifulSoup(html)
table = bsObj.find("table", { "class" : "wikitable sortable" })
n=int(len(table)/2)

results = {}

i=0
for row in table.findAll("tr"):
    
    col = row.findAll("td")
    
    # 3 columns, except the first row which has zero
    if len(col)>1:
        city = col[0].findAll("a")        # get hyperlinks
        descrip = col[0].findAll("small") # get description in case multiple cities
        country= col[1].find("a")         # get hyperlink
        
        # multiple cities, choose 1
        if len(city)>1:
            city = city[findIndex(descrip)]
        else:
            city = city[0]
            
        # attributes of tag object (hyperlink)
        # note: can either use processLink or processLinkByParagraph function
        # processLinkByParagraph was chosen since only interested in data in paragraphs
        results[(city['title'],country['title'])] = {"city": processLinkByParagraph(city['href']), 
                                                     "country": processLinkByParagraph(country['href'])}
        
    i+=1
    print(".....Processing iteration: {}/{}, elapsed time: {}".format(i,n,time.time() - t0)) 

                                                    
#%% Save results #############################################################

# convert tuple key to one string key : capitalName_countryName
output = {k[0] + "_" + k[1] : v for (k,v) in results.items()}        
                                            
with open(out_file_name, 'w') as fp:
    json.dump(output, fp)
                    
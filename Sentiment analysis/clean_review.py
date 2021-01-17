# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:22:55 2021

@author: nishant
"""
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from tqdm import tqdm
import re
import sys


def cleantext(text):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', text)
    
    return cleantext


def cleaned_review(text):
    
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = cleantext(text)
    tokens = tokenizer.tokenize(text)
    new_token = [token for token in tokens if token not in en_stopwords]
    stem_token = [ss.stem(token) for token in new_token]
    clean_text = ' '.join(stem_token)
    
    return clean_text

def getcleandocument(inputfile,outputfile):
    
    out = open(outputfile,'w',encoding='utf8')
    
    with open(inputfile,encoding='utf8') as f :
        reviews = f.readlines()
        
    for review in tqdm(reviews):
        clean_review = cleaned_review(review)
        print((clean_review),file=out)

    out.close()
    
    

if __name__ == '__main__':
    
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    
    tokenizer = RegexpTokenizer(r'\w+')
    ss = SnowballStemmer(language='english')
    en_stopwords = set(stopwords.words('english'))
    
    getcleandocument(inputfile, outputfile)
    
import numpy as np
import pandas as pd
import string
import nltk
import re
from utils import *
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize, sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet
nltk.download('stopwords')
nltk.download('wordnet')
stop_words=set(nltk.corpus.stopwords.words('english'))


def text_preprocessing(text):
    """
    for sentiment analysis task
    - Change negation to full form (e.g."'t" to "not")
    - Correct errors (e.g. add white space after '.')
    - Other normalizalitions
    @param    text (str): a string to be processed
    @return   text (str): the processed string
    """
    text = re.sub(r'(?<=[.,])(?=[^\s])', r' ', text)  # add white space after '.' or ','
    text = re.sub(r'\\', '', text) # Replace '\' with ''
    text = re.sub(r'\_', ' ', text) # Replace '_' with ' '
    text = re.sub(r'\*', '', text) # Replace '*' with ' '
    text = re.sub(r"\'ll", " will", text)  # change 'll to will
    text = re.sub(r"\'t", " not", text) # change 't to not
  #  text = re.sub(r'\s+', ' ', text).strip() # Remove trailing whitespace
    return text 


def mass_text_preprocessing(inlist):
    """ 
    Text preprocessing for a collection of books 
    @param    text(list of strings): each of which is a document representing a book
    @return    text(list of strings): a preprocessed version of the collection
    """
    cleaned_books = []
    for i in range(len(inlist)):
        cleaned_books.append(text_preprocessing(inlist[i]))
    return cleaned_books


def sent_tokenizer(instr):
    """ 
    Sentence segmentation 
    @param    text(str): a document  
    @return    text(list of strings): a list of sentences
    """
    return sent_tokenize(instr)


def mass_sent_tokenizer(inlist): 
    """    
    Sentence segmentation for a list of strings
    @param    text (list of strings): a list of documents
    @return   text (str): nested list of tokenized sentences
    """
    out = []
    for n in range(len(inlist)): 
        book_n = sent_tokenizer(inlist[n])
        out.append(book_n)
    return out


def word_tokenizer(intext):
    """
    Function to perform word tokenization
    
    @param    intext: list of strings (sentences)
    @return   list of tokens for the entire document
    """
    word_tokens = [word_tokenize(t) for t in intext] # tokenize the sentence to word level
    word_tokens = flatten_list(word_tokens)
    return word_tokens

        
def clean_text(txt):
    """ 
    Cleaning data for topic modelling task
    - Lemmatization
    - Turn all words into lower case
    - Remove stop words ("not" and "can")
    @param    text(str): a sentence  
    @return    text(str): a preprocessed sentence
    """
    le = WordNetLemmatizer()
    lowercased = txt.lower()
    word_tokens = word_tokenize(lowercased)
    # remove stop words except 'not' and 'can
    tokens = [le.lemmatize(w) for w in word_tokens if w not in stop_words or w in ['not', 'can']] #and len(w)>3
    cleaned_text = " ".join(tokens)
    return cleaned_text

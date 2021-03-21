#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import *
import string
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn

#new 
from nltk.tokenize.casual import casual_tokenize #casual_tokenize(text, preserve_case=True, reduce_len=False, strip_handles=False)
from nltk.tokenize import TreebankWordTokenizer

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


stemmer = PorterStemmer()
def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    return parsed_text

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    #print(tweet.split())
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()

# Own function
def tokenize_words(tweet, use_stemmer = True):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split(r"[-\s.,;!)]+", tweet.lower())).strip()
    if use_stemmer:
        tokens = [stemmer.stem(t) for t in tweet.split()]
    else:
        tokens = [t for t in tweet.split()]
    return tokens

def pos_tag_tweet(tweet, tokenizer, print_tweet = False):
    tokens = tokenizer(tweet)
    tags = nltk.pos_tag(tokens)
    tag_list = [x[1] for x in tags]
    tag_str = " ".join(tag_list)
    return tag_str


# In[3]:


#Now get other features
sentiment_analyzer = VS()

def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.
    
    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return(parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))

def other_features(tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features"""
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    
    words = preprocess(tweet) #Get text only
    
    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
    num_unique_terms = len(set(words.split()))
    
    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)
    
    twitter_objs = count_twitter_objs(tweet)
    retweet = 0
    if "rt" in words:
        retweet = 1
    features = [FKRA, FRE,syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1],
                twitter_objs[0], retweet]
    #features = pandas.DataFrame(features)
    return features

def get_feature_array(tweets):
    feats=[]
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)


# In[4]:


def print_cm(y,y_preds, save_cm = False, save_path = None):
    plt.rc('pdf', fonttype=42)
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.serif'] = 'Times'
    plt.rcParams['font.family'] = 'serif'
    from sklearn.metrics import confusion_matrix

    confusion_matrix = confusion_matrix(y,y_preds)
    matrix_proportions = np.zeros((3,3))
    for i in range(0,3):
        matrix_proportions[i,:] = confusion_matrix[i,:]/float(confusion_matrix[i,:].sum())
    names=['Hate','Offensive','Neither']
    confusion_df = pd.DataFrame(matrix_proportions, index=names,columns=names)
    plt.figure(figsize=(5,5))
    seaborn.heatmap(confusion_df,annot=True,annot_kws={"size": 12},cmap='gist_gray_r',cbar=False, square=True,fmt='.2f')
    plt.ylabel(r'\textbf{True categories}',fontsize=14)
    plt.xlabel(r'\textbf{Predicted categories}',fontsize=14)
    plt.tick_params(labelsize=12)

    if save_cm:
        if save_path is not None:
            plt.savefig(save_path)
            print(f'Confusionmatrix was saved to {save_path}')
        else:
            save_path = 'data/confusion.png'
            plt.savefig(save_path)
            print(f'Confusionmatrix was saved to {save_path}')

    plt.show()


# In[5]:


# Data Structure
class TweetsDataset:
    def __init__(self, csv_path, tokenizer_name, use_stopwords = True, use_preprocessor= False, min_df = 10, max_df = 0.75, max_ngram = 3):
        
        # Where data is stored
        self.csv_path = csv_path
        
        #Read data directly
        self.dataframe = pd.read_csv(self.csv_path)
        
        # Choose tokenizer
        if tokenizer_name == 'casual_std':
            func = lambda x: casual_tokenize(x, preserve_case=True, reduce_len=False, strip_handles=False)
            self.tokenizer = func
        elif tokenizer_name == 'casual_reduce':
            func = lambda x: casual_tokenize(x, preserve_case=False, reduce_len=True, strip_handles=True)
            self.tokenizer = func
        elif tokenizer_name == 'words':
            self.tokenizer = tokenize_words
        elif tokenizer_name == 'orig':
            self.tokenizer = tokenize
        else:
            raise NotImplementedError('Unknown tokenizer')
            
        # Stopwords
        if use_stopwords:
            self.stopwords = nltk.corpus.stopwords.words("english").extend( ["#ff", "ff", "rt"])
        else:
            self.stopwords = None
            
        # Preprocessor
        if use_preprocessor:
            self.preprocessor = preprocess
        else:
            self.preprocessor = None
        
        
        # Some hyperparameters
        self.min_df = min_df
        self.max_df = max_df
        self.max_ngram = max_ngram
            
        # Vectorizer
        self.vectorizer = TfidfVectorizer(
                        tokenizer=self.tokenizer, #casual_tokenize_specified,
                        preprocessor=self.preprocessor,
                        ngram_range=(1, self.max_ngram),
                        stop_words=self.stopwords,
                        use_idf=True,
                        smooth_idf=False,
                        norm=None,
                        decode_error='replace',
                        max_features=10000,
                        min_df=self.min_df,
                        max_df=self.max_df
                        )
        # PosVectorizer
        self.pos_vectorizer = TfidfVectorizer(
                        tokenizer=None,
                        lowercase=False,
                        preprocessor=None,
                        ngram_range=(1, self.max_ngram),
                        stop_words=None,
                        use_idf=False,
                        smooth_idf=False,
                        norm=None,
                        decode_error='replace',
                        max_features=5000,
                        min_df=5,
                        max_df=0.75,
                        )

        
        #Construct tfidf matrix and get relevant scores
        self.tfidf = self.vectorizer.fit_transform(self.dataframe['tweet']).toarray()
        self.vocab = {v:i for i, v in enumerate(self.vectorizer.get_feature_names())}
        self.idf_vals = self.vectorizer.idf_
        self.idf_dict = {i:self.idf_vals[i] for i in self.vocab.values()}
        print(f'A vocab was created. It consists of {len(self.vocab)} entries')
            
        # POS-tagging
        self.tweet_tags = [pos_tag_tweet(tweet, self.tokenizer, print_tweet = False) for tweet in self.dataframe['tweet']]
        self.pos = self.pos_vectorizer.fit_transform(pd.Series(self.tweet_tags)).toarray()
        self.pos_vocab = {v:i for i, v in enumerate(self.pos_vectorizer.get_feature_names())}
        
        
        # Other features: this is untouched
        self.feats = get_feature_array(self.dataframe['tweet'])
        
        #Now join them all up
        self.features = np.concatenate([self.tfidf,self.pos,self.feats],axis=1)
        self.feature_names = [k for k,_ in self.vocab.items()]+[k for k,_ in self.pos_vocab.items()]+["FKRA", "FRE","num_syllables", "avg_syl_per_word", "num_chars", "num_chars_total",                         "num_terms", "num_words", "num_unique_words", "vader neg","vader pos","vader neu",                         "vader compound", "num_hashtags", "num_mentions", "num_urls", "is_retweet"]
        
        self.labels = self.dataframe['class']
        
        print(f'\n Data has been processed and is now available. Feature dim: {self.features.shape}')






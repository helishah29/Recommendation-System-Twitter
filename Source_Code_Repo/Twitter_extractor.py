#!/usr/bin/env python
# coding: utf-8

# In[58]:


import json
import pandas as pd
import networkx as nx
import re
import sys
import time
import pickle
from urllib.error import URLError
from http.client import BadStatusLine
import twitter

import nltk
from nltk.stem.porter import *
eng_stopwords = nltk.corpus.stopwords.words('english')
#stemmer = PorterStemmer()
from nltk.stem import WordNetLemmatizer   
stemmer = WordNetLemmatizer()


# In[6]:


def oauth_login():
    CONSUMER_KEY = 'Gzy6AjP6DDIbtDWxf94Ucc3Nk'
    CONSUMER_SECRET = 'kRMAj0U87ayefou6dHJfgIsScnCq79mmruRNaztGvpDm8VoY4u'
    auth = twitter.oauth2.OAuth2(CONSUMER_KEY,CONSUMER_SECRET,'AAAAAAAAAAAAAAAAAAAAADc99gAAAAAAzNRZe6QmvPvfKjwAImrzpmQECHc%3DbjCdwhOCbBg0N8TUcywwhi7jpFK1UtSC3M09H0slzo5vIz93mP')
    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api


# ### Make robust request 

# In[7]:



def make_twitter_request(twitter_api_func, max_errors=10, *args, **kw): 
    
    # A nested helper function that handles common HTTPErrors. Return an updated
    # value for wait_period if the problem is a 500 level error. Block until the
    # rate limit is reset if it's a rate limiting issue (429 error). Returns None
    # for 401 and 404 errors, which requires special handling by the caller.
    def handle_twitter_http_error(e, wait_period=2, sleep_when_rate_limited=True):
    
        if wait_period > 3600: # Seconds
            print('Too many retries. Quitting.', file=sys.stderr)
            raise e
    
        # See https://developer.twitter.com/en/docs/basics/response-codes
        # for common codes
    
        if e.e.code == 401:
            print('Encountered 401 Error (Not Authorized)', file=sys.stderr)
            return None
        elif e.e.code == 404:
            print('Encountered 404 Error (Not Found)', file=sys.stderr)
            return None
        elif e.e.code == 429: 
            print('Encountered 429 Error (Rate Limit Exceeded)', file=sys.stderr)
            if sleep_when_rate_limited:
                print("Retrying in 15 minutes...ZzZ...", file=sys.stderr)
                sys.stderr.flush()
                time.sleep(60*15 + 5)
                print('...ZzZ...Awake now and trying again.', file=sys.stderr)
                return 2
            else:
                raise e # Caller must handle the rate limiting issue
        elif e.e.code in (500, 502, 503, 504):
            print('Encountered {0} Error. Retrying in {1} seconds'                  .format(e.e.code, wait_period), file=sys.stderr)
            time.sleep(wait_period)
            wait_period *= 1.5
            return wait_period
        else:
            raise e

    # End of nested helper function
    
    wait_period = 2 
    error_count = 0 

    while True:
        try:
            return twitter_api_func(*args, **kw)
        except twitter.api.TwitterHTTPError as e:
            error_count = 0 
            wait_period = handle_twitter_http_error(e, wait_period)
            if wait_period is None:
                return
        except URLError as e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print("URLError encountered. Continuing.", file=sys.stderr)
            if error_count > max_errors:
                print("Too many consecutive errors...bailing out.", file=sys.stderr)
                raise
        except BadStatusLine as e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print("BadStatusLine encountered. Continuing.", file=sys.stderr)
            if error_count > max_errors:
                print("Too many consecutive errors...bailing out.", file=sys.stderr)
                raise


# In[8]:


def harvest_user_timeline(twitter_api, screen_name=None, user_id=None, max_results=1000):
     
    assert (screen_name != None) != (user_id != None),     "Must have screen_name or user_id, but not both"    
    
    kw = {  # Keyword args for the Twitter API call
        'count': 200,
        'trim_user': 'true',
        'include_rts' : 'true',
        'since_id' : 1
        }
    
    if screen_name:
        kw['screen_name'] = screen_name
    else:
        kw['user_id'] = user_id
        
    max_pages = 16
    results = []
    
    tweets = make_twitter_request(twitter_api.statuses.user_timeline, **kw)
    
    if tweets is None: # 401 (Not Authorized) - Need to bail out on loop entry
        tweets = []
        
    results += tweets
    
    print('Fetched {0} tweets'.format(len(tweets)), file=sys.stderr)
    
    page_num = 1
    
    # Many Twitter accounts have fewer than 200 tweets so you don't want to enter
    # the loop and waste a precious request if max_results = 200.
    
    # Note: Analogous optimizations could be applied inside the loop to try and 
    # save requests. e.g. Don't make a third request if you have 287 tweets out of 
    # a possible 400 tweets after your second request. Twitter does do some 
    # post-filtering on censored and deleted tweets out of batches of 'count', though,
    # so you can't strictly check for the number of results being 200. You might get
    # back 198, for example, and still have many more tweets to go. If you have the
    # total number of tweets for an account (by GET /users/lookup/), then you could 
    # simply use this value as a guide.
    
    if max_results == kw['count']:
        page_num = max_pages # Prevent loop entry
    
    while page_num < max_pages and len(tweets) > 0 and len(results) < max_results:
    
        # Necessary for traversing the timeline in Twitter's v1.1 API:
        # get the next query's max-id parameter to pass in.
        # See https://dev.twitter.com/docs/working-with-timelines.
        kw['max_id'] = min([ tweet['id'] for tweet in tweets]) - 1 
    
        tweets = make_twitter_request(twitter_api.statuses.user_timeline, **kw)
        results += tweets

        print('Fetched {0} tweets'.format(len(tweets)),file=sys.stderr)
    
        page_num += 1
        
    print('Done fetching tweets', file=sys.stderr)

    return results[:max_results]
    
# Sample usage


# In[64]:


#find_url = re.comple(r'^https?:\/\/.*[\r\n]*')
def text_processing(text):
    text1 = re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?|RT|http', '', str(text))
    text2 = re.sub(r'@[A-z0-9]+\b','',text1)
    #text3 = re.sub(r"rt|RT",'',text3)
    return text2


# In[96]:


#text data processing for each person's twitter

def further_processing(text):
    txt = text_processing(text)
    word_lis = [x for x in nltk.word_tokenize(txt) if x.isalpha()]
    word_list = [x.lower() for x in word_lis if x not in eng_stopwords]
    #stemm_list = [stemmer.stem(y.lower()) for y in word_list if y.lower() not in eng_stopwords]
    stemm_list = [stemmer.lemmatize(y.lower()) for y in word_list if y.lower() not in eng_stopwords]
    return ' '.join(stemm_list)

#print(stemm_list)
#with open("louvain_ct_text.pickle",'wb')as f:
#    pickle.dump(word_list,f)


# In[10]:


def get_user_tweets(n,text=None):
    text = []
    twitter_api = oauth_login()
    tweets = harvest_user_timeline(twitter_api, user_id = n ,                                max_results=200)
    for e in tweets:
        b = text_processing(e['text'])   
        text.append(b)
    return {n:text}


# In[82]:


# data = pd.read_csv("data.csv")
# data.head()
# print(data["text"])


# In[83]:


# data["fur_text"] = data["text"].apply(further_processing)
# print(data["fur_text"])


# In[13]:


id_text = {}
n_list = data["id"]
for n in n_list:
    n_dict = get_user_tweets(n)
    id_text.update(n_dict)
    
#print(id_text[836645421971820544][0:10])
#print(id_text[735082409864089601][0:10])
# print(get_user_tweets(n))
    


# In[97]:


with open('user_text.pickle', 'wb') as f:
    result_set = pickle.dump(id_text,f)


# In[98]:


with open('user_text.pickle', 'rb') as f:
    result_set = pickle.load(f)
#print(result_set[836645421971820544])


id_set = []
twitter_set = []

for k,v in result_set.items():
    twitters = " ".join(v)
    text = further_processing(twitters)
    id_set.append(k)
    twitter_set.append(text)

print(twitter_set[0])
   


# In[92]:


db = {"id":id_set,"text":twitter_set}
frame = pd.DataFrame(db)
frame.to_csv('user_text.csv',index=False)


# In[ ]:





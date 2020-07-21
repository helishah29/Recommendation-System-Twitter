#!/usr/bin/env python
# coding: utf-8

# In[175]:


import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# In[176]:


# load the data files
# the users whose text contain nan 
# data1 = pd.read_csv('user_text.csv')
# index = data1['text'].index[data1['text'].apply(pd.isnull)]
# a = data1['id'].iloc[index[:]]

# import data and drop the lines that contains nan 
data  = pd.read_csv('user_text.csv').dropna()
text = data["text"].tolist()
u_text = [str(x) for x in text]
#print(u_text[0])


# # # Vectorize the text files

# In[177]:


#the number of features 
no_features = 2000
# the list of words that is already stemmed
class StemmedVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedVectorizer, self).build_analyzer()
        return lambda doc: ([w.lower() for w in analyzer(doc)])
    
tf_stem_vectorizer = StemmedVectorizer(max_df=0.90,min_df=2, use_idf=True,analyzer="word",max_features=no_features,stop_words = None)
## tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
## tf = tf_stem_vectorizer.fit_transform(u1_topic)
tf = tf_stem_vectorizer.fit_transform(u_text)
tf_feature_names = tf_stem_vectorizer.get_feature_names()


# ## LDA algorithm 
# ### build a topic model, and transform all documents to their topic distrinbutions

# In[178]:


#topic number
no_topics = 10
# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0)
# topic distribution of each document
lda_z = lda.fit_transform(tf)


# In[179]:


print(lda_z.shape)


# In[173]:


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 20


# In[143]:


display_topics(lda, tf_feature_names, no_top_words)
#topics2 = display_topics(lda,tf_feature_names, no_top_words)


# In[174]:


def the_topics(model, feature_names, no_top_words):
    topics = []
    for index , topic in enumerate(model.components_):
        a =  [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topics.extend(a)
    return topics


# In[144]:


# print(lda_z.shape)
# print(lda_z[0])


# In[145]:


main_topic = list(np.argmax(lda_z, axis = 1))


# In[146]:


print(main_topic)


# In[162]:


users = data["id"].tolist()
topic_result = {"user_id":users, "topic":main_topic}
frame = pd.DataFrame(topic_result)
frame.to_csv("topic_result.csv",index = False)

df = pd.read_csv("topic_result.csv")
df.head(50)
df[df["topic"]==1]


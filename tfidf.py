#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[97]:


df=pd.read_csv("C:\\Users\\user\\library_data.csv")


# In[98]:


df


# In[99]:


from sklearn.feature_extraction.text import TfidfVectorizer
corpus = list(df['description'])
for i,item in enumerate(corpus):
    corpus[i]=corpus[i].lower().replace('python',"").replace('library',"").replace('pure',"").replace('package',"")
vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range=(1,1))
X = vectorizer.fit_transform(corpus)


# In[100]:


vectorizer.inverse_transform(X)


# In[101]:


v=vectorizer.transform(['python library to create games'])


# In[102]:


from scipy.sparse import csr_matrix
k=csr_matrix(X).toarray()
max(k[0])


# In[107]:


from scipy import spatial
df1=pd.read_csv("C:\\Users\\user\\Stackoverflow data.csv")
df1=df1[(df1['Type']==1)]
for i,qs in enumerate(list(df1['user1'])):
    v=vectorizer.transform([qs.lower()])
    scores=[]
    for item in k:
        scores.append(1- spatial.distance.cosine(item,csr_matrix(v).toarray()))
    scores=np.array(scores)
    print('QS: '+qs)
    print('LIB: ')
    for item in scores.argsort()[-3:][::-1]:
        print(df['name'][item])
    print()    


# In[63]:





# In[ ]:





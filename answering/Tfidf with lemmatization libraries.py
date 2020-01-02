#!/usr/bin/env python
# coding: utf-8

# In[5]:


import spacy
import pandas as pd
import numpy as np


# In[18]:


df=pd.read_csv("EECE 634\\chatbot\\library_data.csv")
df.drop_duplicates(subset=['name'])


# In[19]:


nlp = spacy.load('en')
def lemmatize_text(input_list):
    lemmatized_descriptions=[]
    for desc in input_list:
        current_desc=[]
        doc = nlp(desc)
        for token in doc:
            current_desc.append(token.lemma_)
        lemmatized_descriptions.append(" ".join(current_desc))
    return lemmatized_descriptions
lemmatized_descriptions=lemmatize_text(list(df['description']))


# In[20]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import stop_words
vectorizer = TfidfVectorizer(stop_words = list(stop_words.ENGLISH_STOP_WORDS)+['python','framework','library','should','import','want','use','pron'], ngram_range=(1,1))
desc_vectors = vectorizer.fit_transform(lemmatized_descriptions)


# In[21]:


vectorizer.inverse_transform(desc_vectors)


# In[22]:


from scipy.sparse import csr_matrix
desc_vectors_arr=csr_matrix(desc_vectors).toarray()

from scipy import spatial
df1=pd.read_csv("data.csv")
lemmatized_qs=lemmatize_text(['library for decision trees']+list(df1[(df1['Type']==1)]['user1']))
for i,qs in enumerate(lemmatized_qs):
    v=vectorizer.transform([qs.lower()])
    isAnswered=0
    print(i+1 ,'-','QS: '+qs)
    if vectorizer.inverse_transform(vectorizer.transform([qs]))[0].shape[0]==0:
        scores=[0]*len(desc_vectors_arr)
    else:
        scores=[]
        for item in desc_vectors_arr:
            scores.append(1- spatial.distance.cosine(item,csr_matrix(v).toarray()))
        scores=np.array(scores)
        print('Ans: ')
        for item in scores.argsort()[-3:][::-1]:
            if scores[item]>0.173:
                if isAnswered:
                    print("Maybe",df['name'][item],"would help")
                else:
                    print(df['name'][item],"is a good choice")
                    isAnswered=1
            elif 0.173>scores[item]>0.129:
                print("I'm not sure, but",df['name'][item],"may help")
                isAnswered=1
            
            
    if isAnswered==0:
        print('Sorry i cannot answer this question yet :)')
    print()    


# In[ ]:





# In[ ]:





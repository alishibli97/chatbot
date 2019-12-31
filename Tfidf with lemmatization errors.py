#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
import pandas as pd
import numpy as np
import re


# In[70]:


df=pd.read_csv("EECE 634\\chatbot\\error_data.csv")


# In[71]:


re.search(r'(?<=named\s)(.)*(?=\s)',"ImportError: No module named <package> blaaaaa".lower()).group(0)


# In[72]:


import re
error_qs=list(df['error'])
for item in error_qs:
    r=re.search(r'(?<=named\s).*',item.lower())
    print(item,r)
    print()


# In[73]:


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
lemmatized_descriptions=lemmatize_text(list(df['error']))


# In[74]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import stop_words
vectorizer = TfidfVectorizer(stop_words = list(stop_words.ENGLISH_STOP_WORDS)+['python','should','want','use','pron'], ngram_range=(1,1))
desc_vectors = vectorizer.fit_transform(lemmatized_descriptions)


# In[80]:


from scipy.sparse import csr_matrix
desc_vectors_arr=csr_matrix(desc_vectors).toarray()

from scipy import spatial
df1=pd.read_csv("data.csv")
orig_qs=['i have error']+list(df1[(df1['Type']==2)]['user1'])
lemmatized_qs=lemmatize_text(orig_qs)
for i,qs in enumerate(lemmatized_qs):
    v=vectorizer.transform([qs.lower()])
    isAnswered=0
    print('QS: '+qs)
    if vectorizer.inverse_transform(vectorizer.transform([qs]))[0].shape[0]==0:
        scores=[0]*len(desc_vectors_arr)
    else:
        scores=[]
        for item in desc_vectors_arr:
            scores.append(1- spatial.distance.cosine(item,csr_matrix(v).toarray()))
        scores=np.array(scores)
        print('Ans: ')
        for item in scores.argsort()[-3:][::-1]:
            if scores[item]>0.3:
                    isAnswered=1
                    if "pip install <package>" in df['how to solve'][item]:
                        try:
                            print(df['how to solve'][item].replace('<package>',re.search(r'(?<=named\s)(.)*?(?=[\s$;,.])',orig_qs[i].lower().replace("'","")).group(0)),scores[item])
                        except:
                            print(df['how to solve'][item],scores[item])
                            
                    else:
                        print(df['how to solve'][item],scores[item])
                        
                    break

            
            
    if isAnswered==0:
        print('Be more specific :)')
    print()    


# In[ ]:





# In[ ]:





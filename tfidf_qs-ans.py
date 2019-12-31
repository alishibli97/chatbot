#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np


df = pd.read_csv("data.csv", encoding = "ISO-8859-1")


from sklearn.feature_extraction.text import TfidfVectorizer
df = df[df['Type'] != 0]
df = df.reset_index(drop=True)
corpus = list(df['user1'])

for i,item in enumerate(corpus):
    corpus[i]=corpus[i].lower().replace('python',"").replace('library',"").replace('pure',"").replace('package',"")
vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range=(1,1))
X = vectorizer.fit_transform(corpus)


vectorizer.inverse_transform(X)


v=vectorizer.transform(['python library to create games'])


from scipy.sparse import csr_matrix
k=csr_matrix(X).toarray()
max(k[0])

correct = 0

from scipy import spatial
df1=pd.read_csv("data_modified.csv", encoding="ISO-8859-1")
df1=df1[df1['modified']==1]
df1 = df1.reset_index(drop=True)
for i,qs in enumerate(list(df1['user1'])):
    v=vectorizer.transform([qs.lower()])
    scores=[]
    for item in k:
        scores.append(1- spatial.distance.cosine(item,csr_matrix(v).toarray()))
    scores=np.array(scores)
    print('QS: '+qs)
    print('Ans: ')
    for item in scores.argsort()[-3:][::-1]:
        print(df['user2'][item])
        if df['user2'][item] == df1['user2'][i]:
            correct += 1
        else:
            print('wrong')
        break
    print()
    print(correct)

print('Accuracy: ' + str(correct/len(df1)))

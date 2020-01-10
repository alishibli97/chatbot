#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction import stop_words


def lemmatize_text(input_list):
    spacy_model = spacy.load('en')
    lemmatized_descriptions = []
    for desc in input_list:
        current_desc = []
        doc = spacy_model(desc)
        for token in doc:
            current_desc.append(token.lemma_)
        lemmatized_descriptions.append(" ".join(current_desc))
    return lemmatized_descriptions


df = pd.read_csv("data.csv", encoding = "ISO-8859-1")


from sklearn.feature_extraction.text import TfidfVectorizer
df = df[df['Type'] != 0]
df = df.reset_index(drop=True)
corpus = list(df['user1'])

# for i,item in enumerate(corpus):
#     corpus[i]=corpus[i].lower()
lemmatized_corpus = lemmatize_text(corpus)
vectorizer = TfidfVectorizer(stop_words = ['a', 'the','python', 'should', 'want', 'use', 'pron'], ngram_range=(1,1))
X = vectorizer.fit_transform(lemmatized_corpus)


vectorizer.inverse_transform(X)


v=vectorizer.transform(['python library to create games'])


from scipy.sparse import csr_matrix
k=csr_matrix(X).toarray()
max(k[0])

correct = 0
correct_1 = 0
correct_2 = 0

from scipy import spatial
df1=pd.read_csv("data_modified.csv", encoding="ISO-8859-1")
df1=df1[df1['modified']==1]
df1 = df1.reset_index(drop=True)
lemmatized_qs = lemmatize_text(list(df1['user1']))
for i,qs in enumerate(lemmatized_qs):
    v=vectorizer.transform([qs.lower()])
    scores=[]
    for item in k:
        scores.append(1- spatial.distance.cosine(item,csr_matrix(v).toarray()))
    scores=np.array(scores)
    print('QS: '+qs)
    print('Ans: ')
    for item in scores.argsort()[-3:][::-1]:
        if scores[item] > 0.45:
            print(df['user2'][item])
            print(scores[item])
            if df['user2'][item] == df1['user2'][i]:
                correct += 1
			correct_1 += 1
            else:
                print('wrong')
            break
        else:
            print('Sorry i cannot answer this question yet :)')
            if df1['user2'][i] == 'Sorry i cannot answer this question yet :)':
                correct += 1
                correct_2 += 1
            else:
                print('wrong')
            break
    
print('Confusion Matrix:')
print("\t\t\tAnswered\tDidn't Answer")
print('Should Answer\t\t' + str(correct_1) + '\t\t' + str(len(df1) - 13 - correct_1))
print("Shouldn't Answer\t" + str(13 - correct_2) + '\t\t' + str(correct_2))
print('')
print('Total Answered Correctly: ' + str(correct) + ' out of ' + str(len(df1)))
print('')
print('Recall: ' + str(correct_1/(len(df1) - 13)))
print('Precision: ' + str(correct_1/(correct_1 + 13 - correct_2)))
print('Accuracy: ' + str(correct/len(df1)))

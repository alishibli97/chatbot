import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = list(set(stopwords.words('english')))

def filterString(str):
    word_tokens = [word.lower() for word in word_tokenize(str)]
    index = [ word for word in word_tokens]
    index = [re.sub(r'\W+','',word) for word in index]
    index = list(filter(None,index))
    return index

df = pd.read_csv("data.csv", encoding = "ISO-8859-1")

k=3
df = df[df['Type'] == k]
df = df.reset_index(drop=True)
corpus = list(df['user1'])

indexes = {} 
i=0
for sentence in corpus:
    indexes[i]=filterString(sentence)
    i+=1

print("Please input exit when you want to exit.")
while True:
    question = input("\nPlease input your question: ")
    if(question!="exit"):
        question = filterString(question)
        min=1.0
        output="Please be more precise."
        indexx=-1
        for index in indexes:
            l = list(set(indexes[index])-set(question))
            percent_diff = len(l)/len(indexes[index])
            if (percent_diff<min and percent_diff<0.5): # and percent_sim<0.65
                #print("Entered")
                #print(percent_sim)
                min=percent_diff
                output=corpus[index]
                indexx=index
        #print("closest index is {} ".format(indexx))
        print("Closest question is: {}\n% difference={}".format(output,min))
    else: break

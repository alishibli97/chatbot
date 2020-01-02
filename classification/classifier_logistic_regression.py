import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import warnings

# ignore warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# read the file
df = pd.read_csv('data.csv', encoding='cp1252')
col = ['Type', 'user1']
df = df[col]

# plot the figure
fig = plt.figure(figsize=(8,6))
df.groupby('Type').user1.count().plot.bar(ylim=0)
#plt.show()

# <=> tf-idf metric in information retrieval for cross-docs
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

# getting features and labels
features = tfidf.fit_transform(df.user1).toarray()
labels = df.Type

# defining the models
model = LogisticRegression(random_state=0)


X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)

model.fit(X_train, y_train)

print("Please input exit when you want to exit.")
while True:
    question = input("\nPlease input your question: ")
    if(question!="exit"):
        x = tfidf.transform([question])
        confidence = max(model.predict_proba(x)[0])
        if(confidence>=0.35):
            print("Category {}, probability: {}".format(model.predict(x)[0],confidence))
        else: print("I don't understand, please be more specific.")


    else: break












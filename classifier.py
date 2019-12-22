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
model1 = LogisticRegression(random_state=0)
model2 = LinearSVC()
model3 = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)
model4 = MultinomialNB()
model5 = KNeighborsClassifier(n_neighbors=3)
model6 = DecisionTreeClassifier(random_state=0)

models = [model1,model2,model3,model4,model5,model6]

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for model in models:
    model_name = model.__class__.__name__
    print("Testing: {}".format(model_name))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    for title, normalize in titles_options:
        # plotting the confusion matrix
        disp = metrics.plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Reds, normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    #plt.show()
    print("\nThe accuracy of {} is {}".format(model_name,metrics.accuracy_score(y_test, y_pred)))
    print(metrics.classification_report(y_test, y_pred))

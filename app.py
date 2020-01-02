from flask import Flask, render_template, request
import numpy as np
from flask import Flask, request
from pymessenger.bot import Bot
import pandas as pd
import spacy
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
from scipy.sparse import csr_matrix
from scipy import spatial
import json
import re

class Chatbot():
    def __init__(self, method=None):
        self.method = method
        self.spacy_model = spacy.load('en')
        self.labels = self.get_labels("annotations.json")
        
        self.library_df=pd.read_csv("library_data.csv")
        self.library_df.drop_duplicates(subset=['name'])
        self.lemmatized_library_descriptions=self.lemmatize_text(list(self.library_df['description']))
        self.library_vectorizer = TfidfVectorizer(stop_words = list(stop_words.ENGLISH_STOP_WORDS)+['a','python','framework','library','should','import','want','use','pron'], ngram_range=(1,1))
        self.library_desc_vectors = self.library_vectorizer.fit_transform(self.lemmatized_library_descriptions)
        self.library_desc_vectors=csr_matrix(self.library_desc_vectors).toarray()
        
        self.error_df=pd.read_csv("C:\\Users\\user\\EECE 634\\chatbot\\error_data.csv")
        self.error_lemmatized_descriptions=self.lemmatize_text(list(self.error_df['error']))
        self.error_vectorizer = TfidfVectorizer(stop_words = list(stop_words.ENGLISH_STOP_WORDS)+['python','should','want','use','pron'], ngram_range=(1,1))
        self.error_desc_vectors = self.error_vectorizer.fit_transform(self.error_lemmatized_descriptions)
        self.error_desc_vectors_arr=csr_matrix(self.error_desc_vectors).toarray()
            
        self.k = []
        self.threshold = [0, 0.5, 0.55, 0.55, 0.5]
        self.vectorizers = []
        self.dff = []
        self.df = pd.read_csv("data.csv", encoding="ISO-8859-1")
        for cat in range(2, 7):
            if cat == 2:  # represents category 0
                vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 1))
                self.vectorizers.append(vectorizer)
                df1 = self.df[self.df['Type'] == 0]
            else:
                vectorizer = TfidfVectorizer(stop_words=['a', 'the', 'python', 'should', 'want', 'use', 'pron'],
                                             ngram_range=(1, 1))
                self.vectorizers.append(vectorizer)
                df1 = self.df[self.df['Type'] == cat]
            df1 = df1.reset_index(drop=True)
            self.dff.append(df1)
            corpus = list(df1['user1'])
            lemmatized_corpus = self.lemmatize_text(corpus)
            X = vectorizer.fit_transform(lemmatized_corpus)
            self.k.append(csr_matrix(X).toarray())

    def lemmatize_text(self,input_list):
        lemmatized_descriptions=[]
        for desc in input_list:
            current_desc=[]
            doc = self.spacy_model(desc)
            for token in doc:
                current_desc.append(token.lemma_)
            lemmatized_descriptions.append(" ".join(current_desc))
        return lemmatized_descriptions
    def get_labels(self, arg):
        with open(arg) as json_file:
            data = json.load(json_file)
            labels = {
                "Greetings": [0, []],
                "Library": [1, []],
                "Error": [2, []],
                "Syntax": [3, []],
                "Interpreted": [4, []],
                "Methods": [5, []],
                "Directory": [6, []]
            }

            for item in data["entities"]:
                value = item["offsets"][0]["text"]
                if (item["classId"] == "e_7"):
                    if value not in labels["Greetings"][1]: labels["Greetings"][1].append(value)
                elif (item["classId"] == "e_8"):
                    if value not in labels["Library"][1]: labels["Library"][1].append(value)
                elif (item["classId"] == "e_9"):
                    if value not in labels["Error"][1]: labels["Error"][1].append(value)
                elif (item["classId"] == "e_10"):
                    if value not in labels["Syntax"][1]: labels["Syntax"][1].append(value)
                elif (item["classId"] == "e_11"):
                    if value not in labels["Interpreted"][1]: labels["Interpreted"][1].append(value)
                elif (item["classId"] == "e_12"):
                    if value not in labels["Methods"][1]: labels["Methods"][1].append(value)
                elif (item["classId"] == "e_13"):
                    if value not in labels["Directory"][1]: labels["Directory"][1].append(value)

            for category in labels:
                txt_file = "features/annotated_" + str(labels[category][0]) + "_" + category + ".txt"
                with open(txt_file, 'w') as file:
                    file.write(json.dumps(labels[category][1]))

            for category in labels:
                txt_file = "features/added_" + str(labels[category][0]) + "_" + category + ".txt"
                with open(txt_file, 'r') as file:
                    x = file.read().splitlines()
                    for value in x:
                        if x not in labels[category][1]: labels[category][1].append(value)
                    file.close()
            return labels
    def answer(self, question, cat):
        if cat==1:
            v=self.library_vectorizer.transform(self.lemmatize_text([question.lower()]))
            isAnswered=0
            if self.library_vectorizer.inverse_transform(self.library_vectorizer.transform(self.lemmatize_text([question.lower()])))[0].shape[0]==0:
                scores=[0]*len(self.library_desc_vectors)
            else:
                scores=[]
                for item in self.library_desc_vectors:
                    scores.append(1- spatial.distance.cosine(item,csr_matrix(v).toarray()))
                scores=np.array(scores)
                answer_list=[]
                for item in scores.argsort()[-3:][::-1]:
                    if scores[item]>0.173:
                        if isAnswered:
                            answer_list.append("Maybe "+self.library_df['name'][item]+" would help")
                        else:
                            answer_list.append(self.library_df['name'][item]+" is a good choice")
                            isAnswered=1
                    elif 0.173>scores[item]>0.129:
                        answer_list.append("I'm not sure, but "+self.library_df['name'][item]+" may help")
                        isAnswered=1
            if isAnswered==0:
                return 'Sorry i cannot answer this question yet :)'
            else:
                return ". ".join(answer_list)
        elif cat==2:
            
            lemmatized_qs=self.lemmatize_text([question])
            for i,qs in enumerate(lemmatized_qs):
                v=self.error_vectorizer.transform([qs.lower()])
                isAnswered=0
                if self.error_vectorizer.inverse_transform(self.error_vectorizer.transform([qs]))[0].shape[0]==0:
                    scores=[0]*len(self.error_desc_vectors_arr)
                else:
                    scores=[]
                    for item in self.error_desc_vectors_arr:
                        scores.append(1- spatial.distance.cosine(item,csr_matrix(v).toarray()))
                    scores=np.array(scores)
                    for item in scores.argsort()[-3:][::-1]:
                        if scores[item]>0.3:
                                isAnswered=1
                                if "pip install <package>" in self.error_df['how to solve'][item]:
                                    try:
                                        return self.error_df['how to solve'][item].replace('<package>',re.search(r'(?<=named\s)(.)*?(?=[\s;,.]*).*$',question.lower().replace("'","")).group(0))
                                    except:
                                        return self.error_df['how to solve'][item]
                                        
                                else:
                                    return self.error_df['how to solve'][item]
                                    
                                break
            
                        
                        
                if isAnswered==0:
                    return 'Be more specific :)'
        else:
            c = 0 if cat == 0 else cat - 2
            lemmatized_qs = self.lemmatize_text([question])
            for i, qs in enumerate(lemmatized_qs):
                v = self.vectorizers[c].transform([qs.lower()])
                scores = []
                for item in self.k[c]:
                    scores.append(1 - spatial.distance.cosine(item, csr_matrix(v).toarray()))
                scores = np.array(scores)
                index = scores.argsort()[-3:][::-1][0]
                if scores[index] > self.threshold[c]:
                    return self.dff[c]['user2'][index],scores[index]
                else:
                    return 'Sorry i cannot answer this question yet :)',[scores[index],self.dff[c]['user2'][index],self.vectorizers[c].inverse_transform(self.vectorizers[c].transform([qs.lower()]))]

    def classify_functional(self, question):
        cat = -1
        cat_found = []
        for category in self.labels:
            for phrase in self.labels[category][1]:
                x = re.search("(^|[^a-zA-Z])" + phrase + "($|[^a-zA-Z])", question, re.IGNORECASE)
                if (x is not None):
                    cat_found.append(category)
                    break
        if (cat_found == []):
            cat = -1
        elif (cat_found == ["Greetings"]):
            cat = 0
        elif (len(cat_found) >= 1):
            if ("Greetings" in cat_found): cat_found.remove("Greetings")
            if (len(cat_found) == 1):
                cat = self.labels[cat_found[0]][0]
            elif ("Error" in cat_found):
                cat = 2
            elif ("Syntax" in cat_found):
                cat = 3
            elif ("Interpreted" in cat_found):
                cat = 4
            elif ("Directory" in cat_found):
                cat = 6
            elif ("Methods" in cat_found):
                cat = 5
            else:
                cat = 1
        if (cat == -1):
            return "I don't understand, please be more specific."
        else:
            return self.answer(question, cat)
        


TEMPLATES_AUTO_RELOAD = True
app = Flask(__name__)
#create chatbot
chatbot = Chatbot()

#define app routes
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/get")
#function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    return str(chatbot.classify_functional(userText))
if __name__ == "__main__":
    
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run()
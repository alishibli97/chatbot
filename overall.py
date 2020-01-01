import tkinter as tk
import json
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
import numpy as np
from scipy import spatial
from scipy.sparse import csr_matrix
import spacy
from sklearn.feature_extraction import stop_words

class Application(tk.Frame):
    def __init__(self, master=None,method=None):
        super().__init__(master)
        self.master = master
        self.method = method
        self.spacy_model = spacy.load('en')
        self.grid(row=0, column=0)
        #self.scrollbar = tk.Scrollbar(orient=tk.VERTICAL)
        #self.scrollbar.grid(row=0, column=1, sticky='ns')
        self.create_widgets()
        self.n = 0
        self.labels = self.get_labels("annotations.json")

        self.library_df = pd.read_csv("library_data.csv")
        self.library_df.drop_duplicates(subset=['name'])
        self.lemmatized_library_descriptions = self.lemmatize_text(list(self.library_df['description']))
        self.library_vectorizer = TfidfVectorizer(
            stop_words=list(stop_words.ENGLISH_STOP_WORDS) + ['a', 'python', 'framework', 'library', 'should', 'import',
                                                              'want', 'use', 'pron'], ngram_range=(1, 1))
        self.library_desc_vectors = self.library_vectorizer.fit_transform(self.lemmatized_library_descriptions)
        self.library_desc_vectors = csr_matrix(self.library_desc_vectors).toarray()

        self.error_df = pd.read_csv("error_data.csv")
        self.error_lemmatized_descriptions = self.lemmatize_text(list(self.error_df['error']))
        self.error_vectorizer = TfidfVectorizer(
            stop_words=list(stop_words.ENGLISH_STOP_WORDS) + ['python', 'should', 'want', 'use', 'pron'],
            ngram_range=(1, 1))
        self.error_desc_vectors = self.error_vectorizer.fit_transform(self.error_lemmatized_descriptions)
        self.error_desc_vectors_arr = csr_matrix(self.error_desc_vectors).toarray()

    # this function is for creating the GUI widgets
    def create_widgets(self):
        self.user = tk.Label(self, text="User").grid(row=0, column=0)
        self.txt = tk.Entry(self, width=140)
        self.txt.grid(row=0, column=1)
        self.send = tk.Button(self, text="Send", command=self.send_message).grid(row=0, column=2)
        self.quit = tk.Button(self, text="Quit", command=self.master.destroy).grid(row=1, column=2)
        self.master.bind('<Return>', self.send_message)

    # this function is triggered on hitting the "send" button or pressing "Enter" from keyboard
    def send_message(self, event=None):
        message = self.retrieve_input()
        if (message != ""):
            self.n += 1
            self.user_name = tk.Label(self, text="User: ").grid(row=self.n, column=0)
            self.user_text = tk.Label(self, text=message, anchor=tk.W, justify=tk.LEFT, width=120).grid(row=self.n,column=1)
            self.reply()

    # this function is for starting the replying procedure
    def reply(self):
        self.n += 1
        self.AI_name = tk.Label(self, text="AI: ", fg='red').grid(row=self.n, column=0)
        self.AI_text = tk.Label(self, text=self.compute_answer(self.retrieve_input()), anchor=tk.W, justify=tk.LEFT,
                                width=120, fg='red', wraplength=700).grid(row=self.n, column=1)
        self.txt.delete(0, 'end')

    # retrives the input from the user
    def retrieve_input(self):
        return self.txt.get()

    # directs search towards particular method
    def compute_answer(self, question):
        if self.method=="functional_functional" or self.method=="functional_databased": return self.classify_functional(question)
        elif self.method=="databased_functional" or self.method=="databased_databased": return self.classify_databased(question)
        else: return 'You had inputted wrong method type: "{}"'.format(self.method)

    # functional model based on regex for classification
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
            if cat==0 or self.method=="functional_functional" or self.method=="databased_functional": return self.answer_functional(question,cat)
            else: return self.answer_databased(question,cat)

    # databased models based (machine learning models and tf-idf measure) for classification
    def classify_databased(self,question):
        warnings.filterwarnings('ignore')

        df = pd.read_csv('data.csv', encoding='cp1252')
        col = ['Type', 'user1']
        df = df[col]

        # <=> tf-idf metric in information retrieval for cross-docs
        # Convert a collection of raw documents to a matrix of TF-IDF features
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english', lowercase=False)

        # getting features and labels
        features = tfidf.fit_transform(df.user1).toarray()
        labels = df.Type

        model = LogisticRegression(random_state=0)

        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,test_size=0.33, random_state=0)
        model.fit(X_train, y_train)

        x = tfidf.transform([question])

        confidence = max(model.predict_proba(x)[0])
        if(confidence>=0):
            return self.answer_functional(question,model.predict(x)[0])

        else: return "I don't understand, please be more specific."

    # functional model for answering (based on Jaccard similarity)
    def answer_functional(self, question, cat):
        file_data = pd.read_csv("data.csv",encoding='ISO-8859-1')
        questions = list(file_data[(file_data['Type'] == cat)]['user1'])
        answers = list(file_data[(file_data['Type'] == cat)]['user2'])

        indexes = {}
        i = 0
        for q in questions:
            if (self.filterString(q) != []): indexes[i] = self.filterString(q)
            i += 1

        question = self.filterString(question)
        min = 0
        threshold = 0.1
        index_target = -1
        for index in indexes:
            # computing Jaccard Similarity [intersection/union]
            inter = set(question).intersection(set(indexes[index]))
            un = set(question).union(indexes[index])
            percent_sim = len(inter) / len(un)
            if (percent_sim > min and percent_sim > threshold):  # and percent_sim<0.65
                min = percent_sim
                output = questions[index]
                index_target = index
        if (index_target != -1):
            return answers[index_target]
        else:
            return "I can't answer this question yet."

    # directing the answering procedure towards 3 main ways
    # first for answering library questions by the help of library descriptions
    # second for answering error questions by the help of error descriptions
    # the third answers for rest of categories 3 4 5 6
    def answer_databased(self,question,cat):
        if(cat==1): return self.answer_databased_library(question,cat)    # answer for library
        elif(cat==2): return self.answer_databased_error(question,cat)      # answer for error
        else: return self.answer_databased_rest(question,cat)               # answer rest of categories

    # answering for library related questions
    def answer_databased_library(self,question,cat):
        v = self.library_vectorizer.transform(self.lemmatize_text([question.lower()]))
        isAnswered = 0
        if self.library_vectorizer.inverse_transform(
                self.library_vectorizer.transform(self.lemmatize_text([question.lower()])))[0].shape[0] == 0:
            scores = [0] * len(self.library_desc_vectors)
        else:
            scores = []
            for item in self.library_desc_vectors:
                scores.append(1 - spatial.distance.cosine(item, csr_matrix(v).toarray()))
            scores = np.array(scores)
            answer_list = []
            for item in scores.argsort()[-3:][::-1]:
                if scores[item] > 0.173:
                    if isAnswered:
                        answer_list.append("Maybe " + self.library_df['name'][item] + " would help")
                    else:
                        answer_list.append(self.library_df['name'][item] + " is a good choice")
                        isAnswered = 1
                elif 0.173 > scores[item] > 0.129:
                    answer_list.append("I'm not sure, but " + self.library_df['name'][item] + " may help")
                    isAnswered = 1
        if isAnswered == 0:
            return 'Sorry i cannot answer this question yet :)'
        else:
             return ". ".join(answer_list)

    # answering for error related questions
    def answer_databased_error(self,question,cat):
        lemmatized_qs = self.lemmatize_text([question])
        for i, qs in enumerate(lemmatized_qs):
            v = self.error_vectorizer.transform([qs.lower()])
            isAnswered = 0
            if self.error_vectorizer.inverse_transform(self.error_vectorizer.transform([qs]))[0].shape[0] == 0:
                 scores = [0] * len(self.error_desc_vectors_arr)
            else:
                scores = []
                for item in self.error_desc_vectors_arr:
                    scores.append(1 - spatial.distance.cosine(item, csr_matrix(v).toarray()))
                scores = np.array(scores)
                for item in scores.argsort()[-3:][::-1]:
                    if scores[item] > 0.3:
                        isAnswered = 1
                        if "pip install <package>" in self.error_df['how to solve'][item]:
                            try:
                                return self.error_df['how to solve'][item].replace('<package>', re.search(
                                    r'(?<=named\s)(.)*?(?=[\s;,.]*).*$', question.lower().replace("'", "")).group(
                                    0))
                            except:
                                   return self.error_df['how to solve'][item]
                        else:
                            return self.error_df['how to solve'][item]
            if isAnswered == 0:
                return 'Be more specific :)'

    # answering for the rest of the categories 3 4 5 6
    def answer_databased_rest(self,question,cat):
        df = pd.read_csv("data.csv", encoding="ISO-8859-1")
        df = df[df['Type'] == cat]
        df = df.reset_index(drop=True)
        corpus = list(df['user1'])

        for i, item in enumerate(corpus):
            corpus[i] = corpus[i].lower().replace('python', "").replace('library', "").replace('pure', "").replace(
                'package', "")
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
        X = vectorizer.fit_transform(corpus)
        k = csr_matrix(X).toarray()

        v = vectorizer.transform([question.lower()])
        scores = []
        for item in k:
            scores.append(1 - spatial.distance.cosine(item, csr_matrix(v).toarray()))
        scores = np.array(scores)
        index = scores.argsort()[-3:][::-1][0]
        return df['user2'][index]

    # filtering the string by lemmatizing and removing non alphanumeric
    def filterString(self, str):
        lemmatizer = WordNetLemmatizer()
        word_tokens = [word.lower() for word in word_tokenize(str)]
        index = [lemmatizer.lemmatize(word) for word in word_tokens]
        index = [re.sub(r'\W+', '', word) for word in index]
        index = list(filter(None, index))
        return index

    # lemmatizing text
    def lemmatize_text(self, input_list):
        lemmatized_descriptions = []
        for desc in input_list:
            current_desc = []
            doc = self.spacy_model(desc)
            for token in doc:
                current_desc.append(token.lemma_)
            lemmatized_descriptions.append(" ".join(current_desc))
        return lemmatized_descriptions

    # getting and reading the labels from the json file for functional classification
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

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", required=False, help="name of the method")
    args = vars(ap.parse_args())

    # default method
    methods =["functional_functional","functional_databased","databased_functional","databased_databased"]
    method = methods[1]
    if (args["name"]!=None): method = args["name"]

    root = tk.Tk()
    root.geometry("1000x500")
    root.title("Python Chatbot: "+method)
    root.resizable(width=False, height=False)
    app = Application(master=root,method=method)
    app.mainloop()

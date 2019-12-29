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

class Application(tk.Frame):
    def __init__(self, master=None,method=None):
        super().__init__(master)
        self.master = master
        self.method = method
        self.grid(row=0, column=0)
        self.create_widgets()
        self.n = 0
        self.labels = self.get_labels("annotations.json")

    def create_widgets(self):
        self.user = tk.Label(self, text="User").grid(row=0, column=0)
        self.txt = tk.Entry(self, width=140)
        self.txt.grid(row=0, column=1)
        self.send = tk.Button(self, text="Send", command=self.send_message).grid(row=0, column=2)
        self.quit = tk.Button(self, text="Quit", command=self.master.destroy).grid(row=1, column=2)
        self.master.bind('<Return>', self.send_message)

    def send_message(self, event=None):
        message = self.retrieve_input()
        if (message != ""):
            self.n += 1
            self.user_name = tk.Label(self, text="User: ").grid(row=self.n, column=0)
            self.user_text = tk.Label(self, text=message, anchor=tk.W, justify=tk.LEFT, width=120).grid(row=self.n,
                                                                                                        column=1)
            self.reply()

    def reply(self):
        self.n += 1
        self.AI_name = tk.Label(self, text="AI: ", fg='red').grid(row=self.n, column=0)
        self.AI_text = tk.Label(self, text=self.compute_answer(self.retrieve_input()), anchor=tk.W, justify=tk.LEFT,
                                width=120, fg='red').grid(row=self.n, column=1)
        self.txt.delete(0, 'end')

    def retrieve_input(self):
        return self.txt.get()

    def compute_answer(self, question):
        if self.method=="functional_functional": return self.classify_functional(question)
        elif self.method=="functional_databased": return self.classify_databased(question)
        else: return "yala min hon"

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
            return self.answer_functional(question, cat)

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

    def answer_functional(self, question, cat):
        with open("data.csv", encoding='unicode_escape') as file:
            file_data = pd.read_csv(file)
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

    def filterString(self, str):
        lemmatizer = WordNetLemmatizer()
        word_tokens = [word.lower() for word in word_tokenize(str)]
        index = [lemmatizer.lemmatize(word) for word in word_tokens]
        index = [re.sub(r'\W+', '', word) for word in index]
        index = list(filter(None, index))
        return index

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
    method = "functional_functional"
    if (args["name"]!=None): method = args["name"]

    root = tk.Tk()
    root.geometry("1000x500")
    root.title("Python Chatbot: "+method)
    root.resizable(width=False, height=False)
    app = Application(master=root,method=method)
    app.mainloop()

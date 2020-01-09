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

def input_user():
    k=0
    df_original = df_original[df_original['Type'] == k]
    df_original = df_original.reset_index(drop=True)
    corpus = list(df_original['user1'])
    answers_original = list(df_original['user2'])
    

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
                if (percent_diff<min and percent_diff<0.5):
                    min=percent_diff
                    output=corpus[index]
                    indexx=index
            print("Closest question is: {}\n% difference={}".format(output,min))
        else: break

def input_file():
    i=0
    
    df = pd.read_csv('data_modified.csv', encoding='cp1252')
    df = df[df["modified"]==1]
    df_questions = df['user1']
    df_answers = list(df['user2'])
    
    correct=0
    
    for question in df_questions:
        k = df.loc[df['user1'] == question]
        k = int(k['Type'])
        
        df_original = pd.read_csv("data.csv", encoding = "ISO-8859-1")
        df_original = df_original[df_original['Type'] == k]
        df_original = df_original.reset_index(drop=True)
        questions_original = list(df_original['user1'])
        answers_original = list(df_original['user2'])

        if k!=1:
            indexes = {} 
            j=0
            for sentence in questions_original:
                indexes[j]=filterString(sentence)
                j+=1
            
            filtered = filterString(question)
            min=0.0
            output="Please be more precise.\n"
            indexx=-1
            for index in indexes:
                intersection = len(list(set(indexes[index]).intersection(filtered)))
                union = (len(indexes[index]) + len(filtered)) - intersection
                jaccard_measure = float(intersection)/union
                if (jaccard_measure>min):
                    min=jaccard_measure
                    output=questions_original[index]
                    indexx=index
            #print(jaccard_measure)
            if indexx==-1 and k==-1:
                #print("YES\n")
                correct+=1
            elif indexx!=-1:
                answer_original = answers_original[indexx]
                answer_predicted = df_answers[i]
                if(answer_predicted==answer_original):
                    #print("YES\n")
                    correct+=1
                else:
                    ok=1
                    #print("% similarity is {}".format(jaccard_measure))
                    #print("My question: "+question)
                    #print("Closest question: "+output+"\n")
            else: ok=1#print(output)
        i+=1
    accuracy = correct/len(df_answers)
    print("Accuracy is: {}".format(accuracy))

if __name__ == '__main__':

    df_original = pd.read_csv("data.csv", encoding = "ISO-8859-1")

        
    #input_user()
    input_file()
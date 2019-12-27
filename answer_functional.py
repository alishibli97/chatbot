import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = list(set(stopwords.words('english')))

def filterString(str):
    word_tokens = [word.lower() for word in word_tokenize(str)]
    index = [ word for word in word_tokens if word not in stop_words ]
    index = [re.sub(r'\W+','',word) for word in index]
    index = list(filter(None,index))
    return index

file_name = "questions_syntax.txt"
#file_name = "questions_method.txt"
#file_name = "questions_dir.txt"

with open(file_name, encoding="utf8") as file:
    f = file.readlines()
    
    indexes = {} 
    i=0
    for sentence in f:
        indexes[i]=filterString(sentence)
        i+=1

    print("Please input exit when you want to exit.")
    while True:
        question = input("\nPlease input your question: ")
        if(question!="exit"):
            question = filterString(question)
            min=1.0
            output="Please be more precise."
            for index in indexes:
                l = list(set(indexes[index])-set(question))
                percent_sim = len(l)/len(indexes[index])
                if (percent_sim<min and percent_sim<0.65):
                    #print("Entered")
                    #print(percent_sim)
                    min=percent_sim
                    output=f[index]
            print("Closest question is: {}\n% difference={}".format(output,min))
        else: break
        
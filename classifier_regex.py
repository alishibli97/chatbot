import json
import re
import pandas as pd

def check_categories(sentence):
    cat_found=[]
    for category in labels:    
        for phrase in labels[category][1]:
            x=re.search("(^|[^a-zA-Z])"+phrase+"($|[^a-zA-Z])",sentence,re.IGNORECASE)
            if(x is not None): 
                cat_found.append(category)
                break
    return cat_found
    
def check_hierarchy(cat_found):
    if(cat_found==[]): return "Please be more precise",-1
    elif(cat_found==["Greetings"]): return "Greetings",0
    elif (len(cat_found)>=1):
        if("Greetings" in cat_found): cat_found.remove("Greetings")
        if(len(cat_found)==1): return cat_found[0],labels[cat_found[0]][0]
        elif("Error" in cat_found): return "Error",2
        elif("Syntax" in cat_found): return "Syntax",3
        elif("Interpreted" in cat_found): return "Interpreted",4
        elif("Directory" in cat_found): return "Directory",6
        elif("Methods" in cat_found): return "Method",5
        else: return 1

def input_user():
    print("\nPlease enter exit to exit.")
    while True:
        sentence=input("\nPlease input your sentence: ")
        if(sentence!="exit"):
            cat_found = check_categories(sentence)
            print("Categories found: ",cat_found)
            name,num=check_hierarchy(cat_found)
            if num==-1: print("Please be more precise.")
            else: print("Belongs to {}, category {}".format(name,num))
        else: break

def input_file():
    df = pd.read_csv('added_questions.csv', encoding='cp1252')
    col = ['Type', 'user1']
    df = df[col]
    
    TP=0 # false positives
    TN=0 # true positives
    FP=0 # false positives
    FN=0 # false negatives
    
    for question in df['user1']:
        cat = df.loc[df['user1'] == question]
        cat = int(cat['Type'])
        
        cat_found=check_categories(question)
        name,num=check_hierarchy(cat_found)
        
        if num==-1:
            if num==cat: TN+=1
            else: FN+=1
        elif num!=-1:
            if num==cat: TP+=1
            else: FP+=1
    
    confusion_matrix = pd.DataFrame([[TN,FP],[FN,TP]])
    
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    
    print("Confusion matrix:\n",confusion_matrix)
    print("Accuary: {}\nRecall: {}\nPrecision: {}".format(accuracy,recall,precision))    
    
with open("annotations.json") as json_file:
    data = json.load(json_file)
    
    # e_7 -> greetings [0] -> label[0]
    # e_8 -> library [1] -> label[1]
    # e_9 -> error [2] -> label[2]
    # e_10 -> syntax [3] -> label[3]
    # e_11 -> interpreted [4] -> label[4]
    # e_12 -> methods [5] -> label[5]
    # e_13 -> directory [6] -> label[6]
    
    labels={
        "Greetings": [0,[]],
        "Library":[1,[]],
        "Error":[2,[]],
        "Syntax":[3,[]],
        "Interpreted":[4,[]],
        "Methods":[5,[]],
        "Directory":[6,[]]
        }
        
    for item in data["entities"]:
        value = item["offsets"][0]["text"]
        if(item["classId"]=="e_7"): 
            if value not in labels["Greetings"][1]: labels["Greetings"][1].append(value)
        elif(item["classId"]=="e_8"):
            if value not in labels["Library"][1]: labels["Library"][1].append(value)
        elif(item["classId"]=="e_9"):
            if value not in labels["Error"][1]: labels["Error"][1].append(value)
        elif(item["classId"]=="e_10"):
            if value not in labels["Syntax"][1]: labels["Syntax"][1].append(value)
        elif(item["classId"]=="e_11"):
            if value not in labels["Interpreted"][1]: labels["Interpreted"][1].append(value)
        elif(item["classId"]=="e_12"):
            if value not in labels["Methods"][1]: labels["Methods"][1].append(value)
        elif (item["classId"]=="e_13"):
            if value not in labels["Directory"][1]: labels["Directory"][1].append(value)
       
    for category in labels.keys():
        txt_file="features/annotated_"+str(labels[category][0])+"_"+category+".txt"
        with open(txt_file,'w') as file:
            file.write(json.dumps(labels[category][1]))
    
    for category in labels.keys():
        txt_file="features/added_"+str(labels[category][0])+"_"+category+".txt"
        with open(txt_file,'r') as file:
            x=file.read().splitlines()
            for value in x:
                if x not in labels[category][1]: labels[category][1].append(value)
            file.close()
            
    #input_user()
    input_file()
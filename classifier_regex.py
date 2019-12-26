import json
import re

with open("annotations.json") as json_file:
    data = json.load(json_file)
    
    # e_7 -> greetings [0] -> label[0]
    # e_8 -> library [1] -> label[1]
    # e_9 -> error [2] -> label[2]
    # e_12 -> methods [5] -> label[3]
    # e_13 -> directory [6] -> label[4]
    
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
    
    print("\nPlease enter exit to exit.")
    while True:
        sentence=input("\nPlease input your sentence: ")
        if(sentence!="exit"):
            cat_found=[]
            for category in labels:    
                for phrase in labels[category][1]:
                    #x=re.search("(^|\s)"+phrase+"($|\s)",sentence,re.IGNORECASE)
                    x=re.search("(^|[^a-zA-Z])"+phrase+"($|[^a-zA-Z])",sentence,re.IGNORECASE)
                    if(x is not None): 
                        cat_found.append(category)
                        break
            print(cat_found)
            if(cat_found==["Greetings"]): print("Belongs to Greetings, category 0.")
            elif (len(cat_found)>=1):
                    if("Greetings" in cat_found): cat_found.remove("Greetings")
                    if(len(cat_found)==1): print("Belongs to {}.".format(cat_found[0]))
                    elif("Error" in cat_found): print("Belongs to Error, category 1.")
                    elif("Syntax" in cat_found): print("Belongs to Syntax, category 2.")
                    elif("Interpreted" in cat_found): print("Belongs to Intepreted, category 3.")
                    elif("Directory" in cat_found): print("Belongs to Directory, category 4.")
                    elif("Methods" in cat_found): print("Belongs to Methods, category 5.")
                    else: print("Belongs to Library, category 6.")
        else: break
        
        
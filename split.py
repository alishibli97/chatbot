import json

path='D:/ali/aub courses/EECE/EECE 634/project/advising.scenario-1.train.json/may30.train.scenario-1.json'
with open(path) as json_file:
    data = json.load(json_file)
    
    f1 = open("subset0.txt","w+")
    f2 = open("subset1.txt","w+")
    f3 = open("subset2.txt","w+")
    
    i=0
    for convo in data:
        print("iteration "+str(i))
        s="dialog "+str(i)+":"
        s+="\n\nA. messages-so-far:"
        for tuple in convo['messages-so-far']:
            s+="\n"+tuple['utterance']
        s+="\n\nB. options-for-correct-answers"
        for tuple in convo['options-for-correct-answers']:
            s+="\n"+tuple['utterance']
        s+="\n\nC. options-for-next:"
        for tuple in convo['options-for-next']:
            s+="\n"+tuple['utterance']
        s+="\n\n-----------------------------------------------------------\n"
        if(i<60):
            f1.write(s)
            i+=1
        elif(i>=60 and i<120):
            f2.write(s)
            i+=1
        elif(i>=120 and i<180):
            f3.write(s)
            i+=1
        else: 
            print("Done.")
            break
import json

with open('D:/ali/aub courses/EECE/EECE 634/project/advising.scenario-1.train.json/may30.train.scenario-1.json') as json_file:
    data = json.load(json_file)
    i=0
    for convo in data:
#        print("message-so far in conversation "+str(i)+":",data['messages-so-far'])
        print("options-for-correct-answers in conversation "+str(i)+":",convo['options-for-correct-answers'])
        print("")
        i+=1

# sample run of last elements
#
# options-for-correct-answers in conversation 99989: [{'candidate-id': 545833110, 'utterance': 'No that'}]
#
#options-for-correct-answers in conversation 99990: [{'candidate-id': 29212879, 'utterance': "Linear Algebra is an advisory prerequisite, which I see you've not taken. I wouldn't recommend taking the course for that reason. You should take MATH 217 first."}]
#
#options-for-correct-answers in conversation 99991: [{'candidate-id': 276903321, 'utterance': 'It does count as a ULCS for Data Science.'}]
#
#options-for-correct-answers in conversation 99992: [{'candidate-id': 863768567, 'utterance': 'It has been rated as more difficult and to have a higher workload by past students'}]
#
#options-for-correct-answers in conversation 99993: [{'candidate-id': 861437360, 'utterance': 'Can i be of any help to you?You are awesome.'}]
#
#options-for-correct-answers in conversation 99994: [{'candidate-id': 354887701, 'utterance': 'is there something specific you are into, I can help you choose the class based on your interest'}]
#
#options-for-correct-answers in conversation 99995: [{'candidate-id': 226342589, 'utterance': 'No, you are cool!'}]
#
#options-for-correct-answers in conversation 99996: [{'candidate-id': 556873135, 'utterance': "I see that you don't have Linear Algebra yet and that's a prerequisite to the course. You should consider taking MATH 217 first."}]
#
#options-for-correct-answers in conversation 99997: [{'candidate-id': 196750942, 'utterance': 'It does count as a ULCS for Data Science.'}]
#
#options-for-correct-answers in conversation 99998: [{'candidate-id': 841503887, 'utterance': 'Past students have rated it as more difficult and having a higher workload.'}]
#
#options-for-correct-answers in conversation 99999: [{'candidate-id': 70208336, 'utterance': "Awesome. You're a great person. Did you need my help with anything else?"}]

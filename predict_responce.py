import nltk
import json
import pickle
import numpy as np
import random
ignore_words = ['?', '!',',','.', "'s", "'m"]
import tensorflow 
from data_preprocessing import get_stem_words
model=tensorflow.keras.models.load_model('chatbot_model.h5')
intents=json.loads(open("intents.json").read())
words=pickle.load(open("words.pkl","rb"))
classes=pickle.load(open("classes.pkl","rb"))
def preprocess_userinput(user_input):
    inputword_token1=nltk.word_tokenize(user_input)
    inputword_token2=get_stem_words(inputword_token1,ignore_words)
    inputword_token2=sorted(list(set(inputword_token2)))
    bag=[]
    bagofwords=[]
    for word in words:
        if word in inputword_token2:
            bagofwords.append(1)
        else:
            bagofwords.append(0)
    bag.append(bagofwords)
    return np.array(bag)
def botclass_prediction(user_input):
    inp=preprocess_userinput(user_input)
    prediction=model.predict(inp)
    predictedclass_label=np.argmax(prediction[0])
    return predictedclass_label
def bot_responce(user_input):
    predictedclass_lable=botclass_prediction(user_input)
    predictedclass=classes[predictedclass_lable]
    for intent in intents["intent"]:
        if intent["tag"]==predictedclass:
            bot_responce=random.choice(intent["responcses"])
            return bot_responce
print("I'm Stella ,How can I help You")
while True:
    user_input=input("Type your message here")
    print(user_input)
    responce=bot_responce(user_input)
    print(responce)


import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
from gtts import gTTS
import os
import time
import speech_recognition as sr

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1  # Corrected from '==' to '='
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break  # Corrected indentation
    return result

print("Hello world")

while True:
    

    #Remove hashtags to use your Microphone
    #recognizer = sr.Recognizer()

    
    #with sr.Microphone() as source:
        #print("Speak something...")
        #audio = recognizer.listen(source)

    #try:
        #print("Recognizing...")
        
        #text = recognizer.recognize_google(audio)
        #print("You said:", text)
    #except sr.UnknownValueError:
        #print("Sorry, I couldn't understand what you said.")
    #except sr.RequestError as e:
        #print("Could not request results; {0}".format(e))




























    #If you enable microphone, make sure to change message variable below
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
    text = res
    language = "en"

    vout = gTTS(text=text, lang=language, slow=False)

    vout.save("response.mp3")
    os.system("play response.mp3")

    




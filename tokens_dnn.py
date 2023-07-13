import os
import pandas as pd
import re
import tensorflow as tf
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

# Preparing the tokenizer
sentences = ["Your sentences", "Another sentence"]  # Change this with your sentences
tokenizer = Tokenizer(num_words=5000, oov_token='OOV')
tokenizer.fit_on_texts(sentences)

# Saving the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Now suppose we're in another script and we load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

# Print the type to make sure it's the correct one
print(type(loaded_tokenizer))

factory = StemmerFactory()  
stemmer = factory.create_stemmer()
key_norm = pd.read_csv('https://raw.githubusercontent.com/kinanti18/ofa-2022/main/key_norm.csv')

model = tf.keras.models.load_model('model/DNN_model.h5')

def text_normalize(text):
    text = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0] if key_norm['singkat'].eq(word).any() else word for word in text.split()])
    text = str.lower(text)
    return text

def stemming(text):
    text = stemmer.stem(text)
    return text

def casefolding(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

def preprocess_text(text):
    text = stemming(text_normalize(casefolding(text)))
    return text

def predict_model_dnn(text):
    maxlen = 1000  # Adjust this to match the input shape your model expects
    X = preprocess_text(text)
    X_pad = pad_sequences(loaded_tokenizer.texts_to_sequences([X]), maxlen=maxlen, dtype='float32')
    output = model.predict(X_pad)
    return output

# Test the prediction function
print(predict_model_dnn("Your test sentence"))

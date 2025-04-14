import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model dan tokenizer
model = load_model("models/chatbot_model.h5")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_length = 20
reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}

def generate_response(user_input):
    sequence = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(sequence, maxlen=max_length, padding="post")
    prediction = model.predict(padded)
    response_index = prediction.argmax()
    return reverse_word_index.get(response_index, "I don't understand")

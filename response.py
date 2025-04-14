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
    padded = pad_sequences(sequence, maxlen=20, padding="post")

    prediction = model.predict(padded)[0]  # Ambil prediksi untuk 1 input

    # Ambil token ID dari hasil prediksi
    predicted_token_ids = prediction.argmax(axis=-1)

    # Decode semua token ke kata
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    response_words = [reverse_word_index.get(i, "") for i in predicted_token_ids if i != 0]

    return " ".join(response_words).strip()
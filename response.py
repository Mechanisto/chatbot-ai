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
    print("Tokenized input:", sequence)

    if not sequence or not sequence[0]:
        return "Sorry, I didn't understand that input."

    padded = pad_sequences(sequence, maxlen=20, padding="post")
    prediction = model.predict(padded)[0]

    predicted_token_id = np.argmax(prediction)
    print("Predicted token ID:", predicted_token_id)

    word = reverse_word_index.get(predicted_token_id, "")
    return word.strip() if word else "I don't understand"
    print(generate_response("hello"))   # atau kata dari kolom "sentence"

import pickle
import os
from model import create_model
from preprocess import load_data
from sklearn.model_selection import train_test_split
x, y, tokenizer= load_data()

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
model = create_model(vocab_size=5000, max_length=20)
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_val, y_val))
model.save("models/chatbot_model.h5")
# Simpan model
os.makedirs("models", exist_ok=True)
model.save("models/chatbot_model.h5")

# Simpan tokenizer
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
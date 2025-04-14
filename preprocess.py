import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(file_path="data/dataSet.xlsx", num_words=5000, max_len=20):
    # Baca dataset
    data = pd.read_excel(file_path)
    sentences = data["sentence"].astype(str).tolist()
    responses = data["response"].astype(str).tolist()
    
    # Gabungkan semua teks untuk melatih tokenizer
    all_text = sentences + responses
    all_text = [text.lower() for text in all_text]

    # Tokenizer
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(all_text)

    # Input (X)
    input_sequences = tokenizer.texts_to_sequences(sentences)
    padded_input = pad_sequences(input_sequences, maxlen=max_len, padding="post")

    # Output (y)
    output_sequences = tokenizer.texts_to_sequences(responses)
    padded_output = pad_sequences(output_sequences, maxlen=max_len, padding="post")

    # Buat dimensi y jadi (samples, timesteps, 1)
    padded_output = np.expand_dims(padded_output, axis=-1)

    return padded_input, padded_output, tokenizer

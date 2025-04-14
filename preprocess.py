import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(file_path="data\dataSet.xlsx", num_words=5000,max_len=20):
    # Contoh dataset sederhana
    data = pd.read_excel(file_path)
    sentences = data["sentence"].astype(str).tolist()
    responses = data["response"].astype(str).tolist()
    
    all_text= sentences+responses
    all_text = [text.lower() for text in all_text]
    
    # Tokenize the texts
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(all_text)

    input_sequences = tokenizer.texts_to_sequences(sentences)
    padded_input = pad_sequences(input_sequences, maxlen=max_len, padding="post")
    
    output_sequences = tokenizer.texts_to_sequences(responses)
    padded_output = [seq[0] if len(seq) > 0 else 0 for seq in output_sequences]
    padded_output=np.array(padded_output)

    return padded_input, padded_output, tokenizer

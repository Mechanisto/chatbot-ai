from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed

def create_model(vocab_size=5000, max_length=20):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
        LSTM(256, return_sequences=True),
        LSTM(256, return_sequences=True),
        TimeDistributed(Dense(vocab_size, activation='softmax'))
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

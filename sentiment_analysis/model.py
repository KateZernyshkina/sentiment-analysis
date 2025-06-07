import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding, Input


def build_model(vocab_size, embedding_dim, max_length, lstm_units, dropout, num_classes):
    input_layer = Input(shape=(max_length,))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)
    x = Bidirectional(LSTM(lstm_units[0], return_sequences=True))(embedding_layer)
    x = Dropout(dropout[0])(x)
    x = LSTM(lstm_units[1], return_sequences=True)(x)
    x = Dropout(dropout[1])(x)
    x = LSTM(lstm_units[2])(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(dropout[2])(x)
    output_layer = Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

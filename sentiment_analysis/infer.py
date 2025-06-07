import pickle

import numpy as np
import tensorflow as tf
from src.preprocess import clean_text
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_artifacts():
    model = tf.keras.models.load_model("model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder


def predict_sentiment(comment, model, tokenizer, label_encoder, max_length):
    comment = clean_text(comment)
    seq = tokenizer.texts_to_sequences([comment])
    padded = pad_sequences(seq, maxlen=max_length, padding="post", truncating="post")
    prediction = model.predict(padded)[0]
    return label_encoder.inverse_transform([np.argmax(prediction)])[0]

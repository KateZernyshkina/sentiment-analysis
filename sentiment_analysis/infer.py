import pickle

import hydra
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sentiment_analysis.preprocess import clean_text


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


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig):
    print("Give me a comment :)")
    comment = input()
    model, tokenizer, label_encoder = load_artifacts()
    print("This is ", predict_sentiment(comment, model, tokenizer, label_encoder, cfg.model.max_length), " comment.")


if __name__ == "__main__":
    main()

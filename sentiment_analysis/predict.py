import pickle
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sentiment_analysis.preprocess import clean_text


def load_artifacts(model_path="model.keras", tokenizer_path="tokenizer.pkl", label_encoder_path="label_encoder.pkl"):
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder


def predict_sentiment(texts, model, tokenizer, label_encoder, max_length):
    cleaned_texts = [clean_text(text) for text in texts]
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")
    preds = model.predict(padded)
    pred_labels = label_encoder.inverse_transform(preds.argmax(axis=1))
    return pred_labels


def plot_and_save_confusion_matrix(y_true, y_pred, label_encoder, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)  # Создаём папку, если нет
    conf_matrix = confusion_matrix(y_true, y_pred, labels=label_encoder.classes_)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig):
    csv_path = Path("data/market_comments.csv")
    df = pd.read_csv(csv_path)
    texts = df["comment"].astype(str).tolist()

    model, tokenizer, label_encoder = load_artifacts()
    max_length = cfg.model.max_length

    pred_labels = predict_sentiment(texts, model, tokenizer, label_encoder, max_length)
    df["predicted_tonality"] = pred_labels

    if "tonality" in df.columns:
        true_labels = df["tonality"].values
        plot_and_save_confusion_matrix(
            true_labels, pred_labels, label_encoder, save_path=Path("plots/confusion_matrix.png")
        )
    else:
        print("Колонка 'tonality' отсутствует — матрица ошибок не построена.")

    print(df[["comment", "predicted_tonality"]].head())
    output_path = Path("data/predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    return df


if __name__ == "__main__":
    main()

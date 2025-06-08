import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


def preprocess_text(text):
    text = text.lower()
    words = re.findall(r"\b\w+\b", text)
    return set(words)


def build_word_stats(df, text_col="comment", label_col="tonality"):
    pos_texts = df[df[label_col] == "positive"]
    neg_texts = df[df[label_col] == "negative"]
    n_pos = len(pos_texts)
    n_neg = len(neg_texts)

    word_pos_counts = defaultdict(int)
    word_neg_counts = defaultdict(int)

    for text in pos_texts[text_col]:
        words = preprocess_text(text)
        for w in words:
            word_pos_counts[w] += 1

    for text in neg_texts[text_col]:
        words = preprocess_text(text)
        for w in words:
            word_neg_counts[w] += 1

    word_pos_freq = {w: count / n_pos for w, count in word_pos_counts.items()}
    word_neg_freq = {w: count / n_neg for w, count in word_neg_counts.items()}

    return word_pos_freq, word_neg_freq


def predict_sentiment(texts, word_pos_freq, word_neg_freq):
    predictions = []
    for text in texts:
        words = preprocess_text(text)
        neg_score = sum(word_neg_freq.get(w, 0) for w in words)
        pos_score = sum(word_pos_freq.get(w, 0) for w in words)
        if neg_score > pos_score:
            predictions.append("negative")
        else:
            predictions.append("positive")
    return predictions


def plot_confusion_matrix(y_true, y_pred, labels, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)  # Создаёт папку plots, если нет
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5, linecolor="gray")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def main():
    data_path = Path("data/market_comments.csv")
    df = pd.read_csv(data_path)
    if "tonality" not in df.columns:
        raise ValueError("В CSV отсутствует колонка 'tonality' с истинными метками.")

    word_pos_freq, word_neg_freq = build_word_stats(df)
    texts = df["comment"].astype(str).tolist()
    true_labels = df["tonality"].tolist()

    preds = predict_sentiment(texts, word_pos_freq, word_neg_freq)
    df["predicted_tonality"] = preds

    labels = ["positive", "negative"]
    plot_confusion_matrix(true_labels, preds, labels, save_path=Path("plots/confusion_matrix_basic.png"))

    print(df[["comment", "tonality", "predicted_tonality"]].head(20))
    df.to_csv(data_path.parent / "predictions_basic.csv", index=False)
    print(f"Predictions saved to {data_path.parent / 'predictions_basic.csv'}")
    accuracy = accuracy_score(true_labels, preds)
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()

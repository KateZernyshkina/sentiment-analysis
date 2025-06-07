import hydra
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from sentiment_analysis.data import download_data
from sentiment_analysis.model import build_model
from sentiment_analysis.preprocess import clean_text


def save_metric_plot(history, metric_name, val_metric_name, filename):
    plt.figure()
    plt.plot(history.history[metric_name], label=f"train_{metric_name}")
    plt.plot(history.history[val_metric_name], label=f"val_{val_metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} over epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    print("Hello!")
    # Set up MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    # mlflow.create_experiment(cfg.mlflow.experiment_name)
    # Download data
    df = download_data().dropna()
    print(df.head())
    comments = df["comment"].astype(str).apply(clean_text).values
    labels = df["tonality"].values
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    categorical_labels = to_categorical(encoded_labels)

    # Tokenization
    tokenizer = Tokenizer(num_words=cfg.model.vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(comments)
    sequences = tokenizer.texts_to_sequences(comments)
    data_padded = pad_sequences(sequences, maxlen=cfg.model.max_length, padding="post", truncating="post")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data_padded,
        categorical_labels,
        test_size=cfg.train.validation_split,
        random_state=cfg.train.random_seed,
    )

    # Build and train model
    model = build_model(
        vocab_size=cfg.model.vocab_size,
        embedding_dim=cfg.model.embedding_dim,
        max_length=cfg.model.max_length,
        lstm_units=cfg.model.lstm_units,
        dropout=cfg.model.dropout,
        num_classes=categorical_labels.shape[1],
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    mlflow.tensorflow.autolog()
    with mlflow.start_run():
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
        )
        save_metric_plot(history, "accuracy", "val_accuracy", "plots/accuracy.png")
        save_metric_plot(history, "loss", "val_loss", "plots/loss.png")
        # Save model and tokenizer
        model.save("model.keras")
        import pickle

        with open("tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)


if __name__ == "__main__":
    main()

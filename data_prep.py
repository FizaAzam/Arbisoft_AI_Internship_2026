import re

from datasets import load_dataset


def clean_text(text):
    text = re.sub(r"<[^>]*>", " ", text)
    text= re.sub(r"https?://\S+", " ", text)
    text= re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text= text.strip()
    return text


def prepare_frame(df):
    prepared = df.copy()
    prepared["text"] = prepared["text"].str.lower()
    prepared["text"] = prepared["text"].apply(clean_text)
    return prepared


def load_imdb():
    dataset = load_dataset("stanfordnlp/imdb")
    train_df = prepare_frame(dataset["train"].to_pandas())
    test_df = prepare_frame(dataset["test"].to_pandas())
    return train_df, test_df

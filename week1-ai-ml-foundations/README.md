# IMDB Sentiment Classifier

A sentiment classifier for IMDB movie reviews: TF-IDF features and logistic
regression, trained on the 25,000-review Stanford IMDB training split and
evaluated on the held-out 25,000-review test split.

## Results

| Metric | Score |
| --- | --- |
| Test accuracy | 88% |
| Precision / recall / F1 | ~0.88 across both classes |

The confusion matrix is close to symmetric (1,520 false positives against 1,480
false negatives), so the model is not biased toward either sentiment.

## Approach

1. **Load** the `stanfordnlp/imdb` dataset via HuggingFace `datasets`, which
   ships with `train` / `test` splits already separated.
2. **Clean** each review: lowercase, strip HTML tags (the raw reviews contain
   `<br />`), strip URLs, drop punctuation, collapse repeated whitespace.
3. **Vectorize** with `TfidfVectorizer(stop_words="english")`. The vectorizer is
   fit on the training split only and then applied to the test split, so no test
   vocabulary or IDF statistics leak into training.
4. **Train** a `LogisticRegression` classifier — a strong, fast baseline for
   high-dimensional sparse text features.
5. **Evaluate** with accuracy, a classification report, and a confusion matrix.

No separate validation split is used: the dataset ships with a test split, and
with a single model at default settings there is no hyperparameter search that
would require one.

## Project structure

```
data_prep.py            cleaning and loading utilities
main.py                 training and evaluation pipeline
tests/test_data_prep.py unit tests for the cleaning utilities
docs/PROMPT_LOG.md      development log
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

```bash
uv run python main.py
```

The first run downloads the IMDB dataset (~80 MB) and caches it locally.
Training takes well under a minute on CPU.

## Development

```bash
uv run pytest        # run the test suite
uv run ruff check .  # lint
```

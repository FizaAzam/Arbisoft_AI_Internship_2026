# Prompt Log — Sentiment Classifier (IMDB)

**Task:** Train a simple classifier on a public dataset (chose IMDB movie reviews for sentiment).
**Tools:** Python, HuggingFace `datasets`, pandas, scikit-learn.

---

## 1. Planning the pipeline
**Prompt:** Asked for an explanation of the full ML process — download → preprocessing → split → evaluate.
**Outcome / learning:** Understood the standard flow: load data → clean text → vectorize → split (train/test/validation) → train classifier → evaluate. Learned that for text, preprocessing has two parts: cleaning *and* converting words to numbers.

## 2. Loading the dataset
**Prompt:** How to download/import the dataset.
**Outcome / learning:** Used `load_dataset("stanfordnlp/imdb")`. Learned that `datasets` v5 requires the full `owner/name` path (bare `"imdb"` no longer works). Dataset comes pre-split into `train` and `test`, each row `{text, label}` with label 0=negative, 1=positive.

## 3. Environment / setup issues
**Notes:** Hit a stuck Jupyter kernel while the dataset was downloading; resolved by killing the kernel processes and reloading. Installed `datasets` into the project virtualenv (pandas/scikit-learn/numpy were already present).

Environment setup itself:
1. `uv init .` — creates the readme, main file, `pyproject.toml`, and `.python-version`.
2. `uv add numpy pandas scikit-learn datasets` — also creates the virtualenv folder.
3. `.\.venv\Scripts\activate` — activates the virtualenv.
4. `python main.py` — runs the pipeline.

## 4. Preprocessing — cleaning
**Prompt:** How to lowercase the whole column, then remove HTML tags, URLs, punctuation, and extra spaces.
**Outcome / learning:**
- Lowercasing a whole column uses `.str.lower()` (not `.lower()`, which only works on a single string).
- Built a `clean_text()` function using the `re` module:
  - `<[^>]*>` → remove HTML tags (e.g. `<br />`)
  - `https?://\S+` → remove URLs
  - `[^\w\s]` → remove punctuation/symbols (keep only word chars + whitespace)
  - `\s+` → collapse extra whitespace, then `.strip()`
- Learned to apply a function across a column with `.apply(clean_text)`, and to assign back to `df["text"]` (not `df`) to preserve the `label` column.
- Understood `r"..."` raw strings for regex, and that `^` means "start of string" outside `[]` but "not" inside `[]`.

## 5. Vectorizing (text → numbers)
**Prompt:** What is TF-IDF, and how does scikit-learn do it.
**Outcome / learning:**
- TF-IDF = Term Frequency × Inverse Document Frequency: scores words high when frequent in a review but rare across all reviews.
- Used `TfidfVectorizer(stop_words='english')` — the `stop_words` param handles stop-word removal automatically.
- Learned the `fit`/`transform` pattern: **fit on train only**, transform both train and test, to avoid data leakage.
- `fit_transform` on train, `transform` on test — using ONE saved vectorizer object so both share the same vocabulary/IDF.

## 6. Splitting / validation
**Prompt:** Do we need a validation set?
**Outcome / learning:** IMDB already ships train/test splits. For a single simple model with default settings, a separate validation set isn't required. It's only needed when tuning/comparing models (to keep the test set as an honest final exam).

## 7. Training the classifier
**Prompt:** Why Logistic Regression.
**Outcome / learning:**
- Chose `LogisticRegression` — fast, handles high-dimensional sparse TF-IDF data well, interpretable, strong baseline (~88–90% on IMDB).
- Learned that data goes to `.fit(X, y)`, NOT the constructor; constructor takes settings only.
- `X_train` = TF-IDF feature matrix (from vectorizer); `y_train` = the `label` column (came with the data, just renamed).
- Imports must be specific: `from sklearn.linear_model import LogisticRegression` (top-level `import sklearn` doesn't expose submodules).

## 8. Evaluation
**Prompt:** How to evaluate / calculate error.
**Outcome / learning:**
- Predict with `model.predict(X_test)`, then compare to `Y_test`.
- Used `accuracy_score`, `classification_report`, and `confusion_matrix`.
- **Result: 88% accuracy**, balanced across both classes.
- Confusion matrix: 10980 TN, 11020 TP, 1520 FP, 1480 FN — errors roughly equal, so no class bias.

## 9. Testing and project layout
**Prompt:** Write unit tests for the data-prep functions.
**Outcome / learning:** Importing a module runs it top to bottom, so `clean_text` had to move out of `main.py` (which downloads the dataset and trains at import time) into `data_prep.py`. `main.py` now guards its pipeline behind `if __name__ == "__main__":`, which is only true when a file is run directly and not when it is imported. Tests import `data_prep` alone and run without touching the network.

---

## Result summary
Final model: TF-IDF (with English stop words) + Logistic Regression on IMDB → **~88% test accuracy**, balanced precision/recall/F1 across positive and negative classes.

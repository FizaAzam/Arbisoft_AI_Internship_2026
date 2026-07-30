from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_prep import load_imdb


def main():
    train_df, test_df = load_imdb()

    vectorizer = TfidfVectorizer(stop_words='english')

    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    Y_train= train_df["label"]
    Y_test= test_df["label"]

    model=LogisticRegression()
    model.fit(X_train, Y_train)

    Y_predict= model.predict(X_test)

    accuracy=accuracy_score(Y_test, Y_predict)
    print ("Accuracy score: ",accuracy)
    print(classification_report(Y_test, Y_predict))
    print("confusion matrix:\n", confusion_matrix(Y_test, Y_predict))


if __name__ == "__main__":
    main()

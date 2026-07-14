import re

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

dataset=load_dataset("stanfordnlp/imdb")

#print(dataset["train"][0:3])

train_df = dataset["train"].to_pandas()
test_df= dataset["test"].to_pandas()

train_df["text"]= train_df["text"].str.lower()
test_df["text"]= test_df["text"].str.lower()#.str for whole column

def clean_text(text):
    text = re.sub(r"<[^>]*>", " ", text)
    text= re.sub(r"https?://\S+", " ", text)
    text= re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text= text.strip()
    return text

train_df["text"]=train_df["text"].apply(clean_text) #.apply() for every element of col
test_df["text"]=test_df["text"].apply(clean_text)


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


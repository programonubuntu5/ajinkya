
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

messages = [
    "Win free money now",
    "Claim your prize now",
    "Free offer click here",
    "Win a free gift",
    "Get money quickly",
    "Meeting at college today",
    "Please send your notes",
    "Let us study together",
    "Class starts tomorrow",
    "Please call me later"
]

labels = [
    "Spam", "Spam", "Spam", "Spam", "Spam",
    "Ham", "Ham", "Ham", "Ham", "Ham"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

model = MultinomialNB()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

msg = ["Win free prize"]
new_msg = vectorizer.transform(msg)
print("Message:", msg[0])
print("Result:", model.predict(new_msg)[0])

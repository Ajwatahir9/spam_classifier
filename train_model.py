import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. LOAD DATASET (Kaggle file)
df = pd.read_csv("spam.csv")

# 2. CLEAN DATA (VERY IMPORTANT)
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# 3. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# 4. VECTORIZE TEXT
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)

# 5. TRAIN MODEL
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. SAVE MODEL + VECTORIZER
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved successfully ")
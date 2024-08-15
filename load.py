import sqlite3
import nltk
from joblib import load
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import sys


# Load the saved model
loaded_classifier = load('svm_model.joblib')

# Preprocessing function
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Removing stopwords and non-alphanumeric tokens
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    
    return ' '.join(stemmed_tokens)

# Initialize SQLite database connection and cursor
conn = sqlite3.connect('predictions.db')
cursor = conn.cursor()

# Create table to store predictions if not exists
cursor.execute('''CREATE TABLE IF NOT EXISTS predictions
                  (input TEXT, prediction TEXT)''')

# Read tweet from stdin
tweet = sys.stdin.read()

# Preprocess the input tweet
cleaned_input_tweet = preprocess_text(tweet)

# Load the count vectorizer and tfidf transformer
count_vectorizer = load('count_vectorizer.joblib')
tfidf_transformer = load('tfidf_transformer.joblib')

# Vectorization and transformation
X_new_counts = count_vectorizer.transform([cleaned_input_tweet])
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# Make predictions
prediction = loaded_classifier.predict(X_new_tfidf)

# Map prediction to sentiment label
sentiment_label = ""
if prediction[0] == 0:
    sentiment_label = "Neutral"
elif prediction[0] == -1:
    sentiment_label = "Negative"
else:
    sentiment_label = "Positive"

# Store prediction in the database
cursor.execute("INSERT INTO predictions (input, prediction) VALUES (?, ?)", (tweet, sentiment_label))
conn.commit()

# Close the database connection
conn.close()

# Return prediction
print(sentiment_label)

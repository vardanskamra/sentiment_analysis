import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
data = pd.read_csv('merged_data.csv', header=None, skiprows=1, encoding='ISO-8859-1')

data[0] = data[0].astype(str)
data.dropna(inplace=True)

#data1=data.head(8000)
#data2=data.tail(8000)
#data=pd.concat([data1, data2])

print(data.head())
print(data.tail())

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

# Preprocess the tweet text
data['clean_text_2'] = data[0].apply(preprocess_text)

# Splitting into features and labels
X = data['clean_text_2']
y = data[1]

# Vectorization and transformation
count_vectorizer = CountVectorizer()
X_counts = count_vectorizer.fit_transform(X)

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Initialize SVM classifier
classifier = SVC(C=100, gamma=0.1, verbose=1)

# Training the classifier
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the trained model
dump(classifier, 'svm_model.joblib')

# Save the fitted CountVectorizer
dump(count_vectorizer, 'count_vectorizer.joblib')

# Save the fitted TfidfTransformer
dump(tfidf_transformer, 'tfidf_transformer.joblib')

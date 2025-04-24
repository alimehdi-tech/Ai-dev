import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample text
text = "Ali is learning wordpress on online resorses!!"

# Tokenize the text and convert to lowercase
tokens = word_tokenize(text.lower())  

# Perform regex to remove punctuation
tokens_without_punctuation = [word for word in tokens if re.match(r'^[a-zA-Z]+$', word)]

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatize the tokens
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens_without_punctuation]

# Remove stopwords
stop_words = set(stopwords.words('english'))
tokens_without_stopwords = [word for word in lemmatized_tokens if word not in stop_words]

print(tokens_without_stopwords)

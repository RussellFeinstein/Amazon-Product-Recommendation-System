import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

_STOP_WORDS = set(stopwords.words('english'))
_PUNCT_SET = set(string.punctuation)
_LEMMATIZER = WordNetLemmatizer()


def remove_stop_words(tokens):
    return [w for w in tokens if w not in _STOP_WORDS]


def remove_punctuation(tokens):
    filtered = [''.join(c for c in s if c not in _PUNCT_SET) for s in tokens]
    return [s for s in filtered if s]


def lemmatize_tokens(tokens):
    return [_LEMMATIZER.lemmatize(s) for s in tokens]


def join_tokens(tokens):
    return ' '.join(tokens)

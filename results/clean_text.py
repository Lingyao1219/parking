import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()

def remove_punctuations(text):
    """Remove punctuations from text"""
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def remove_digits(text):
    """Remove digits from text"""
    text = ''.join([digit for digit in text if not digit.isdigit()])
    return text

def remove_stopwords(text):
    """Remove stopwords from text"""
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
    return text

def nltk2wn_tag(nltk_tag):
    """Convert nltk tag to wordnet tag"""
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None
    
def lemmatize_text(text):
    """Lemmatize text"""
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(text))  
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:            
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))
    return " ".join(res_words)

def clean_text(text):
    """Clean text by apply multiple text processing steps"""
    text = text.lower()
    text = remove_punctuations(text)
    text = remove_digits(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text
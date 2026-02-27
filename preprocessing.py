import re, nltk
from nltk.stem import WordNetLemmatizer

nltk_ready = False
wnl = WordNetLemmatizer()

def ensure_nltk():
    global nltk_ready
    if nltk_ready:
        return 
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk_ready = True
    
def preprocess_base(text):
    if not isinstance(text, str):
        return ""
    
    #oddzielam znaki interpunkcji spacja + znak + spacja
    text = re.sub(r"([!?.;,])", r" \1 ", text)

    #Usuwam wszystko oprócz tego co jest podane w []
    text = re.sub(r"[^a-zA-Z0-9!?'.,; ]+", "", text)

    #Zamieniam więcej spacji na jedną
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    return text.strip()

def preprocess_vector(text):
    ensure_nltk()
    text = preprocess_base(text)

    #usunięcie wielokrotnych znaków interpunkcyjnych
    text = re.sub(r"([!?])\1{1,}", r"\1", text)

    words = text.split()
    #Sprowadzanie do lematu, podstawowej formy
    words_l = [wnl.lemmatize(w) for w in words]

    return " ".join(words_l)
    
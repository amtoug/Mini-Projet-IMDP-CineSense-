from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import spacy
import joblib

class Vectorisation:
    def __init__(self, ngrams=(1, 2)):
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=["ner", "parser", "textcat"]
        )
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngrams,
            max_features=20000
        )

    def lemmatize(self, text: str) -> str:
        doc = self.nlp(text)
        return " ".join(
            t.lemma_
            for t in doc
            if not t.is_stop and not t.is_punct and t.is_alpha
        )

    def fit_transform(self, X):
        X = X.apply(self.lemmatize)
        return self.vectorizer.fit_transform(X)

    def transform(self, X):
        X = X.apply(self.lemmatize)
        return self.vectorizer.transform(X)
    

vectorisation = Vectorisation(ngrams=(1, 2))
joblib.dump(vectorisation, "vectorizer.pkl")

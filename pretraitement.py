from joblib import load
import re, nltk
nltk.download("stopwords")
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer

model = load('new_model.joblib')
vectorizer = load('vectoriz.joblib')


def cleaned_text(text: str):
    text = str(text)
    clean = re.sub(r"\n", " ", text)
    clean = clean.lower()
    clean = re.sub(r"[~.,%/:;?_&+*=!-]", " ", clean)
    clean = re.sub(r"[^a-z]", " ", clean)
    clean = clean.strip()
    clean = re.sub(r"\s{2,}", " ", clean)
    return clean


def vectorize_text_column(text_column):
   
    # Initialisation du vectoriseur TF-IDF
    vectorizer = TfidfVectorizer()
    # Appliquer le vectoriseur sur la colonne de texte
    features = vectorizer.fit_transform(text_column)
    # Convertir les caractéristiques en une représentation de matrice dense
    features_dense = features.toarray()
    return features_dense

def analyze_sentiment(text):
    # Traduire le texte en anglais
    translated_text = GoogleTranslator(source="fr", target="en").translate(text)
    
    # Nettoyer le texte
    clean_text = cleaned_text(translated_text)
    
    # Vectoriser le texte
    text_vector = vectorizer.transform([clean_text])
    
    # Prédire le sentiment avec le modèle de régression logistique
    prediction = model.predict(text_vector)
    
    return "Positive" if prediction == 1 else "Negative"
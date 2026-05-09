import re
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):

    # Lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Process text
    doc = nlp(text)

    cleaned_words = []

    for token in doc:

        # Keep important supportive words
        if not token.is_stop or token.text in ['not', 'no', 'never', 'but']:

            if not token.is_punct:

                cleaned_words.append(token.lemma_)

    return " ".join(cleaned_words)
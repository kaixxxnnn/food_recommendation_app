import spacy
from typing import Dict, List


class NLPProcessor:
    def __init__(self, model_path: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_path)
        except OSError:
            print(f"Model {model_path} not found. Please install it using:")
            print(f"python -m spacy download {model_path}")
            raise

        self.food_keywords = {
            'cuisine': ['italian', 'chinese', 'indian', 'thai', 'japanese', 'american', 'korean', 'malaysian'],
            'dietary': ['vegetarian', 'vegan', 'gluten-free', 'dairy-free', 'halal'],
            'category': ['breakfast', 'lunch', 'dinner', 'dessert', 'appetizer'],
            'flavor': ['spicy', 'sweet', 'sour', 'bitter', 'savory', 'umami']
        }

    def extract_preferences(self, text: str) -> Dict[str, List[str]]:
        text = text.lower().strip()
        doc = self.nlp(text)

        preferences = {key: [] for key in self.food_keywords.keys()}

        for token in doc:
            if not token.is_stop and not token.is_punct:
                lemma = token.lemma_.lower()
                for category, keywords in self.food_keywords.items():
                    if lemma in keywords:
                        preferences[category].append(lemma)

        for ent in doc.ents:
            word = ent.text.lower()
            for category, keywords in self.food_keywords.items():
                if any(keyword in word for keyword in keywords):
                    preferences[category].append(word)

        for category in preferences:
            preferences[category] = list(set(preferences[category]))

        return preferences

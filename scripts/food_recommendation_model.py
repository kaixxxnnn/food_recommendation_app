import spacy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Dict, Optional
import warnings
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')
import pickle
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

class FoodRecommendationModel:
    def __init__(self, model_path: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_path)
        except OSError:
            raise RuntimeError(f"spaCy model {model_path} not found. Install it with: python -m spacy download {model_path}")

        self.decision_tree = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'food_id'

        self.food_df = None
        self.users_df = None
        self.interactions_df = None

        self.food_keywords = {
            'cuisine': ['italian', 'chinese', 'indian', 'thai', 'japanese', 'american', 'korean', 'malaysian'],
            'dietary': ['vegetarian', 'vegan', 'gluten-free', 'dairy-free', 'halal'],
            'category': ['breakfast', 'lunch', 'dinner', 'dessert', 'appetizer'],
            'flavor': ['spicy', 'sweet', 'sour', 'bitter', 'savory', 'umami']
        }

    def load_datasets(self, food_path: str, users_path: str, interactions_path: str):
        try:
            self.food_df = pd.read_csv(food_path).fillna('unknown')
            self.users_df = pd.read_csv(users_path).fillna('unknown')
            self.interactions_df = pd.read_csv(interactions_path).fillna(0)
            return True
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return False

    def prepare_training_data(self) -> pd.DataFrame:
        if any(df is None for df in [self.food_df, self.users_df, self.interactions_df]):
            raise ValueError("All datasets must be loaded first")

        training_data = []
        for _, interaction in self.interactions_df.iterrows():
            user_id = interaction['user_id']
            food_id = interaction['food_id']
            rating = interaction['rating']

            user_features = self.create_user_features(user_id)
            food_features = self.create_food_features(food_id)
            interaction_features = self.calculate_interaction_features(user_id, food_id)

            sample = {
                'user_age': user_features['user_age'],
                'user_type': user_features['user_type'],
                'preferred_cuisines': user_features.get('preferred_cuisines', []),
                'dietary_restrictions': user_features.get('dietary_restrictions', []),
                'health_conscious': user_features['health_conscious'],
                'adventurous_score': user_features['adventurous_score'],
                'food_cuisine': food_features.get('food_cuisine', 'unknown'),
                'food_category': food_features.get('food_category', 'unknown'),
                'food_dietary_tags': food_features.get('food_dietary_tags', []),
                'food_flavor_profile': food_features.get('food_flavor_profile', []),
                'food_calories': food_features.get('food_calories', 0),
                'food_popularity': food_features.get('food_popularity', 0),
                'avg_rating': interaction_features['avg_rating'],
                'interaction_count': interaction_features['interaction_count'],
                'cuisine_familiarity': interaction_features['cuisine_familiarity'],
                'food_id': food_id,
                'will_like': 1 if rating >= 4 else 0
            }

            for field in ['preferred_cuisines', 'dietary_restrictions', 'food_dietary_tags', 'food_flavor_profile']:
                if isinstance(sample[field], str):
                    import ast
                    try:
                        sample[field] = ast.literal_eval(sample[field])
                    except:
                        sample[field] = []

            training_data.append(sample)

        return pd.DataFrame(training_data)


    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        encoded_df = df.copy()

        single_cat_cols = ['user_type', 'food_cuisine', 'food_category']
        multi_label_cols = ['preferred_cuisines', 'dietary_restrictions', 'food_dietary_tags', 'food_flavor_profile']

        # 1. One-hot encode single-category columns
        for col in single_cat_cols:
            if col in encoded_df.columns:
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                reshaped = encoded_df[[col]].fillna("unknown")
                transformed = ohe.fit_transform(reshaped)
                col_names = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                ohe_df = pd.DataFrame(transformed, columns=col_names, index=encoded_df.index)
                encoded_df = encoded_df.drop(columns=col)
                encoded_df = pd.concat([encoded_df, ohe_df], axis=1)
                self.label_encoders[col] = ohe

        # 2. Multi-label binarize list-like columns
        for col in multi_label_cols:
            if col in encoded_df.columns:
                mlb = MultiLabelBinarizer()

                def parse_values(val):
                    if isinstance(val, list):
                        return val
                    elif pd.isna(val) or val == '' or val == '[]':
                        return []
                    elif isinstance(val, str) and val.startswith('['):
                        import ast
                        try:
                            return ast.literal_eval(val)
                        except:
                            return []
                    else:
                        return [v.strip() for v in str(val).split(',') if v.strip()]

                parsed = encoded_df[col].apply(parse_values)
                transformed = mlb.fit_transform(parsed)
                mlb_df = pd.DataFrame(transformed, columns=[f"{col}_{cls}" for cls in mlb.classes_], index=encoded_df.index)
                encoded_df = encoded_df.drop(columns=col)
                encoded_df = pd.concat([encoded_df, mlb_df], axis=1)
                self.label_encoders[col] = mlb

        return encoded_df

    def train_model(self, test_size: float = 0.2):
        df = self.encode_features(self.prepare_training_data())
        
        y = df['will_like']
        df = df.drop(columns=['will_like'])

        num_cols = ['user_age', 'health_conscious', 'adventurous_score',
                    'food_calories', 'food_popularity', 'avg_rating',
                    'interaction_count', 'cuisine_familiarity']
        num_cols = [col for col in num_cols if col in df.columns] 

        df[num_cols] = self.scaler.fit_transform(df[num_cols])

        self.feature_columns = df.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=test_size, stratify=y, random_state=42)

        self.logis_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        self.logis_reg.fit(X_train.values, y_train)

        y_pred = self.logis_reg.predict(X_test.values)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"✅ Logistic Regression trained.")
        print(f"Accuracy:  {acc:.4f}")

        self.save_model("model\\model.pkl")

        return acc, y_test, y_pred
    
    def create_user_features(self, user_id: Optional[int] = None) -> Dict:
        if user_id is None or self.users_df is None:
            return {
                'user_age': 30,
                'user_type': 'regular',
                'preferred_cuisines': [],
                'dietary_restrictions': [],
                'health_conscious': 0,
                'adventurous_score': 0.5
            }

        row = self.users_df[self.users_df['user_id'] == user_id]
        if row.empty:
            return self.create_user_features()
        user = row.iloc[0]
        return {
            'user_age': user['age'],
            'user_type': user['user_type'],
            'preferred_cuisines': user['preferred_cuisines'],
            'dietary_restrictions': user['dietary_restrictions'],
            'health_conscious': user['health_conscious'],
            'adventurous_score': user['adventurous_score']
        }

    def create_food_features(self, food_id: int) -> Dict:
        if self.food_df is None:
            return {}
        row = self.food_df[self.food_df['food_id'] == food_id]
        if row.empty:
            return {}
        food = row.iloc[0]
        return {
            'food_cuisine': food['cuisine'],
            'food_category': food['category'],
            'food_dietary_tags': food['dietary_tags'],
            'food_flavor_profile': food['flavor_profile'],
            'food_calories': food['calories'],
            'food_popularity': food['popularity_score'],
            'name': food.get('name', 'Unnamed Dish'),
            'ingredients': food.get('ingredients', [])
        }

    def calculate_interaction_features(self, user_id: int, food_id: int) -> Dict:
        if self.interactions_df is None:
            return {'avg_rating': 0, 'interaction_count': 0, 'cuisine_familiarity': 0}

        user_data = self.interactions_df[self.interactions_df['user_id'] == user_id]
        food_info = self.create_food_features(food_id)
        avg_rating = user_data['rating'].mean() if not user_data.empty else 0
        interaction_count = len(user_data)
        cuisine_match = user_data[user_data['food_cuisine'] == food_info.get('food_cuisine', 'unknown')]
        cuisine_familiarity = len(cuisine_match)
        return {
            'avg_rating': avg_rating,
            'interaction_count': interaction_count,
            'cuisine_familiarity': cuisine_familiarity
        }

    def create_feature_vector(self, user_features: Dict, food_features: Dict, interaction_features: Dict) -> pd.DataFrame:
        fv = {
            'user_age': user_features['user_age'],
            'health_conscious': user_features['health_conscious'],
            'adventurous_score': user_features['adventurous_score'],
            'food_calories': food_features.get('food_calories', 0),
            'food_popularity': food_features.get('food_popularity', 0),
            'avg_rating': interaction_features['avg_rating'],
            'interaction_count': interaction_features['interaction_count'],
            'cuisine_familiarity': interaction_features['cuisine_familiarity']
        }

        for col in ['user_type', 'food_cuisine', 'food_category']:
            if col in self.label_encoders:
                ohe = self.label_encoders[col]
                val = [[user_features.get(col)] if col in user_features else [food_features.get(col)]]
                onehot = ohe.transform(val)
                onehot_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                fv.update(dict(zip(onehot_cols, onehot[0])))

        for col in ['preferred_cuisines', 'dietary_restrictions', 'food_dietary_tags', 'food_flavor_profile']:
            if col in self.label_encoders:
                mlb = self.label_encoders[col]
                raw = user_features.get(col, []) if 'user' in col else food_features.get(col, [])
                transformed = mlb.transform([raw])
                mlb_cols = [f"{col}_{cls}" for cls in mlb.classes_]
                fv.update(dict(zip(mlb_cols, transformed[0])))

        fv_df = pd.DataFrame([fv])

        for col in self.feature_columns:
            if col not in fv_df.columns:
                fv_df[col] = 0
        fv_df = fv_df[self.feature_columns]  # Reorder columns to match training

        numeric_cols = ['user_age', 'health_conscious', 'adventurous_score',
                        'food_calories', 'food_popularity', 'avg_rating',
                        'interaction_count', 'cuisine_familiarity']
        numeric_cols = [col for col in numeric_cols if col in fv_df.columns]
        fv_df[numeric_cols] = self.scaler.transform(fv_df[numeric_cols])

        return fv_df

    def save_model(self, path="model\\model.pkl"):
        with open(path, "wb") as f:
            pickle.dump({
                "classifier": self.logis_reg,
                "scaler": self.scaler,
                "label_encoders": self.label_encoders,
                "feature_columns": self.feature_columns
            }, f)
        print(f"✅ Model saved to {path}")

    def load_model(self, path="model\\model.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.logis_reg = data["classifier"]
            self.scaler = data["scaler"]
            self.label_encoders = data["label_encoders"]
            self.feature_columns = data["feature_columns"]
        print(f"✅ Model loaded from {path}")


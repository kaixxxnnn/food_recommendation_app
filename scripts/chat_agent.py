from nlp_processor import NLPProcessor
from recommendation_utils import recommend_top_n_filtered

class FoodChatAgent:
    def __init__(self, model):
        self.model = model
        self.nlp_processor = NLPProcessor()

    def filter_foods(self, preferences):
        if self.model.food_df is None:
            raise ValueError("Food dataset is not loaded.")

        df = self.model.food_df.copy()
        for col, val in preferences.items():
            if val:
                if col == 'dietary':
                    df = df[df['dietary_tags'].str.contains('|'.join(val), case=False, na=False)]
                elif col == 'flavor':
                    df = df[df['flavor_profile'].str.contains('|'.join(val), case=False, na=False)]
                elif col == 'category':
                    df = df[df['category'].str.contains('|'.join(val), case=False, na=False)]
                elif col == 'cuisine':
                    df = df[df['cuisine'].str.contains('|'.join(val), case=False, na=False)]
        return df

    def recommend(self, user_input, user_profile, N=5):
        print("\U0001f440 Understanding your craving...")
        preferences = self.nlp_processor.extract_preferences(user_input)
        print(f"\U0001f50d Detected preferences: {', '.join([f'{k}: {', '.join(v)}' for k, v in preferences.items() if v]) or 'none'}")

        filtered_foods = self.filter_foods(preferences)
        if filtered_foods.empty:
            print("\U0001f625 Sorry, I couldn't find any food that matches your request.")
            return None

        print(f"\U0001f37d️ Searching through {len(filtered_foods)} matching food options...")
        recs = recommend_top_n_filtered(self.model, user_profile, filtered_foods['food_id'].tolist(), N)
        if recs.empty:
            print("\U0001f613 No strong matches found, but I’ll learn from this!")
            return None

        print("\n\U0001f31f Here are some delicious options you might enjoy:")
        for i, row in recs.iterrows():
            print(f"    \U0001f37d️  {row.get('name', 'Unnamed Dish')} — *{row['category'].title()}* ({row['cuisine'].title()})")
            print(f"    \U0001f525 Flavor: {row.get('flavor_profile', 'unknown').title()} | \U0001f3f7️ Dietary: {row.get('dietary_tags', 'unknown').title()}")
            print(f"    \U0001f522 Calories: {int(row.get('calories', 0))} | \U0001f44d Likelihood of Liking: {row['like_probability']*100:.1f}%")
            print(f"    \U0001f4c8 Ingredients: {row.get('ingredients', 'unknown')}")
            print("")
        return recs

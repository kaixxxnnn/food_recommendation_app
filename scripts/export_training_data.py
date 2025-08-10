'''Exports the training data for model evaluation in SAS VIya.'''

from food_recommendation_model import FoodRecommendationModel

model = FoodRecommendationModel()
model.load_datasets("data\\foods.csv", "data\\users.csv", "data\\interactions.csv")

df = model.prepare_training_data()
df = model.encode_features(df)
df.to_csv("data\\training_data.csv", index=False)



from chat_agent import FoodChatAgent
from food_recommendation_model import FoodRecommendationModel

model = FoodRecommendationModel()
model.load_datasets("data\\foods.csv", "data\\users.csv", "data\\interactions.csv")
model.train_model() # or model.load_model("model\\model.pkl")

agent = FoodChatAgent(model)

user_profile = {
    'user_age': 24,
    'user_type': 'adventurous',
    'preferred_cuisines': ['chinese'],
    'dietary_restrictions': ['gluten-free'],
    'health_conscious': 0.1,
    'adventurous_score': 0.8
}

agent.recommend("I want spicy korean food", user_profile, N=5)

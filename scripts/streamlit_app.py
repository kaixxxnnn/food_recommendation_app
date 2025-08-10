import streamlit as st
from chat_agent import FoodChatAgent
from food_recommendation_model import FoodRecommendationModel

st.set_page_config(page_title="🍜 AI Food Recommender", layout="wide")

st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-size: 18px !important;
        }
        h1 {
            font-size: 40px !important;
        }
        h3 {
            font-size: 28px !important;
        }
        input, select, textarea {
            font-size: 18px !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1>🍱 AI Food Recommender</h1>", unsafe_allow_html=True)
st.markdown("<h3>Tell me what you want to eat and I’ll recommend something yummy!</h3>", unsafe_allow_html=True)

# --- Load model ---
@st.cache_resource
def load_model():
    model = FoodRecommendationModel()
    model.load_datasets("data\\foods.csv", "data\\users.csv", "data\\interactions.csv")
    model.load_model("model\\model.pkl") #or train model if not exists
    return model

model = load_model()
agent = FoodChatAgent(model)

user_types = sorted(model.users_df['user_type'].dropna().unique().tolist())

# --- User Input ---
st.subheader("💬 What are you craving?")
user_input = st.text_input("E.g. 'I want spicy halal chinese dinner'", "")

st.subheader("👤 Your Profile")
age = st.number_input("Age", min_value=10, max_value=100, value=24)
user_type = st.selectbox("User Type", user_types)
preferred_cuisine = st.multiselect("Preferred Cuisine (Optional)", [
    "chinese", "italian", "japanese", "thai", "indian", "american", "malaysian"])
dietary = st.multiselect("Dietary Restriction (Optional)", [
    "none", "vegetarian", "vegan", "gluten-free", "halal", "dairy-free"])
health_conscious = st.slider("Health Consciousness", 0.0, 1.0, 0.1)
adventurous_score = st.slider("Adventurousness", 0.0, 1.0, 0.8)

# --- User Profile Packaging ---
user_profile = {
    'user_age': age,
    'user_type': user_type,
    'preferred_cuisines': preferred_cuisine,
    'dietary_restrictions': dietary,
    'health_conscious': health_conscious,
    'adventurous_score': adventurous_score
}

# --- Recommendation Output ---
if st.button("🍽️ Get Recommendations") and user_input:
    recs = agent.recommend(user_input, user_profile, N=5)
    if recs is not None and not recs.empty:
        st.success("✅ Recommendations found!")
        st.subheader("🌟 Here are some delicious options you might enjoy:")

        for idx, row in recs.iterrows():
            st.markdown(f"""
            <div style="font-size: 18px; background-color:#f9f9f9;padding:20px 25px;margin:15px 0;
                        border-radius:12px;border:1px solid #ccc;">
                <h4 style="margin-bottom:8px">🍽️ <b>{row['name']}</b> — <em>{row['category'].title()}</em> ({row['cuisine'].title()})</h4>
                <p style="margin:5px 0 0">
                    🔥 <b>Flavor:</b> {row.get('flavor_profile', 'N/A')}<br>
                    🧂 <b>Ingredients:</b> {row.get('ingredients', 'N/A')}<br>
                    🏷️ <b>Dietary:</b> {row.get('dietary_tags', 'N/A')}<br>
                    🔢 <b>Calories:</b> {int(row.get('calories', 0))}<br>
                    👍 <b>Likelihood of Liking:</b> {row['like_probability']*100:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ No matching foods found.")

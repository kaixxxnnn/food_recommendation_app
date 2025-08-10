import pandas as pd
import ast

def safe_join_list(val):
    if isinstance(val, list):
        return ', '.join(val)
    elif isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return ', '.join(parsed)
        except:
            return val  # it's just a string, return as is
    return str(val)

def recommend_top_n_filtered(model, user_profile, food_ids, N=5):
    recommendations = []

    for food_id in food_ids:
        food_features = model.create_food_features(food_id)
        interaction_features = model.calculate_interaction_features(
            user_id=user_profile.get("user_id", -1),
            food_id=food_id
        )
        fv = model.create_feature_vector(user_profile, food_features, interaction_features)

        prob = model.logis_reg.predict_proba(fv.values)[0][1]

        recommendations.append({
            "food_id": food_id,
            "like_probability": prob,
            "name": food_features.get("name", "Unnamed Dish"),
            "category": food_features.get("food_category", "unknown"),
            "cuisine": food_features.get("food_cuisine", "unknown"),
            "flavor_profile": safe_join_list(food_features.get("food_flavor_profile", [])),
            "dietary_tags": safe_join_list(food_features.get("food_dietary_tags", [])),
            "ingredients": safe_join_list(food_features.get("ingredients", [])),
            "calories": food_features.get("food_calories", 0)
        })

    recs_df = pd.DataFrame(recommendations)
    recs_df = recs_df.sort_values(by="like_probability", ascending=False).head(N).reset_index(drop=True)

    recs_df = recs_df.drop_duplicates(subset='name', keep='first')
    return recs_df

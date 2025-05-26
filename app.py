from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import random
# Load saved models and encoders
model = joblib.load("nutrition_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder_dict = joblib.load("encoders.pkl")
meals_data = joblib.load("meals_dataset.pkl")  # Fix meals_data path
  # Load meal dataset

app = Flask(__name__)
CORS(app,  resources={r"/predict": {"origins": "*"}})
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    if request.content_type != "application/json":
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json()

    try:
        # Extract input data
        age = data["age"]
        gender = data["gender"]
        weight_kg = data["weight_kg"]
        height_cm = data["height_cm"]
        activity_level = data["activity_level"]
        health_condition = data["health_condition"]
        goal = data["goal"]

        # Encode categorical inputs
        gender_encoded = encoder_dict["Gender"].transform([gender])[0]
        activity_level_encoded = encoder_dict["Activity_Level"].transform([activity_level])[0]
        health_condition_encoded = encoder_dict["Health_Condition"].transform([health_condition])[0]

        # Prepare input array (no dietary_restriction as input)
        user_data = np.array([[age, gender_encoded, weight_kg, height_cm, activity_level_encoded, health_condition_encoded]])
        user_data_scaled = scaler.transform(user_data)

        # Predict all targets including dietary_restriction_encoded
        predictions = model.predict(user_data_scaled)[0]

        if len(predictions) != 6:
            return jsonify({"error": "Unexpected output size from model"}), 500

        calories_burned, target_calories, protein, carbs, fat, dietary_restriction_encoded = predictions

        # Round and cast dietary_restriction_encoded to nearest int (since regression output)
        dietary_restriction_encoded = int(round(dietary_restriction_encoded))
        # Clamp to valid classes range (optional safety)
        dietary_restriction_encoded = max(0, min(dietary_restriction_encoded, len(encoder_dict["Dietary_Restriction"].classes_) - 1))

        # Decode dietary restriction string label
        dietary_restriction = encoder_dict["Dietary_Restriction"].inverse_transform([dietary_restriction_encoded])[0]

        # Adjust target calories by goal
        if goal.lower() == "lose weight":
            target_calories -= 500
        elif goal.lower() == "gain weight":
            target_calories += 500

        # Calculate exercise time
        base_time = {"Sedentary": 30, "Lightly Active": 45, "Moderately Active": 60, "Very Active": 90}
        health_modifier = {"None": 1.0, "Diabetes": 1.2, "Hypertension": 1.1, "Obesity": 1.5}
        activity_time = base_time.get(activity_level, 45)
        modifier = health_modifier.get(health_condition, 1.0)
        exercise_time = int(activity_time * modifier + max(0, (target_calories - calories_burned) // 50))

        # Filter meals by encoded dietary restriction (since meals_data is encoded)
        dietary_restriction_encoded_for_filter = encoder_dict["Dietary_Restriction"].transform([dietary_restriction])[0]
        filtered_meals = meals_data[meals_data["Dietary_Restriction"] == dietary_restriction_encoded_for_filter]

        # Fallback if no meals found
        if filtered_meals.empty:
            # Try 'General' or 0 or whatever default class you have
            fallback_encoded = 0
            filtered_meals = meals_data[meals_data["Dietary_Restriction"] == fallback_encoded]
            if filtered_meals.empty:
                return jsonify({"error": "No meal plans available for your dietary restriction or fallback."}), 404

        # Sample 7 random meals for a weekly plan
        weekly_meal_plan = []
        sampled_meals = filtered_meals.sample(n=7, replace=True)
        for day, meal in enumerate(sampled_meals.itertuples()):
            weekly_meal_plan.append({
                "day": f"Day {day + 1}",
                "breakfast": meal.Breakfast,
                "lunch": meal.Lunch,
                "dinner": meal.Dinner,
                "snacks": meal.Snacks,
            })

        # Prepare response
        response = {
            "calories_burned": round(calories_burned, 2),
            "target_calories": round(target_calories, 2),
            "protein_g": round(protein, 2),
            "carbs_g": round(carbs, 2),
            "fat_g": round(fat, 2),
            "exercise_time_minutes": exercise_time,
            "dietary_restriction": dietary_restriction,
            "weekly_meal_plan": weekly_meal_plan
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

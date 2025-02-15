from flask import Flask, request, jsonify
import numpy as np
import joblib
import random

# Load saved models and encoders
model = joblib.load("nutrition_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder_dict = joblib.load("encoder_dict.pkl")
meals_data = joblib.load("meals_dataset.pkl")  # Load meal dataset

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    try:
        age = data["age"]
        gender = data["gender"]
        weight_kg = data["weight_kg"]
        height_cm = data["height_cm"]
        activity_level = data["activity_level"]
        health_condition = data["health_condition"]
        goal = data["goal"]
        
        # Encode categorical values
        gender_encoded = encoder_dict["Gender"].transform([gender])[0]
        activity_level_encoded = encoder_dict["Activity_Level"].transform([activity_level])[0]
        health_condition_encoded = encoder_dict["Health_Condition"].transform([health_condition])[0]
        
        # Create input array
        user_data = np.array([[age, gender_encoded, weight_kg, height_cm, activity_level_encoded, health_condition_encoded]])
        user_data_scaled = scaler.transform(user_data)
        
        # Predict values
        predictions = model.predict(user_data_scaled)[0]
        calories_burned, target_calories, protein, carbs, fat, dietary_restriction_encoded = predictions
        
        # Adjust target calories based on goal
        if goal.lower() == "lose weight":
            target_calories -= 500
        elif goal.lower() == "gain weight":
            target_calories += 500
        
        # Calculate recommended exercise time
        base_time = {"Sedentary": 30, "Lightly Active": 45, "Moderately Active": 60, "Very Active": 90}
        health_modifier = {"None": 1.0, "Diabetes": 1.2, "Hypertension": 1.1, "Obesity": 1.5}
        activity_time = base_time.get(activity_level, 45)
        modifier = health_modifier.get(health_condition, 1.0)
        exercise_time = int(activity_time * modifier + max(0, (target_calories - calories_burned) // 50))
        
        # Decode dietary restriction
        dietary_restriction = encoder_dict["Dietary_Restriction"].inverse_transform([int(round(dietary_restriction_encoded))])[0]
        
        # Generate a weekly meal plan
        meals = meals_data[meals_data["Dietary_Restriction"] == dietary_restriction]
        weekly_meal_plan = []
        for day in range(7):
            daily_meal = meals.sample(n=1).iloc[0]
            weekly_meal_plan.append({
                "day": f"Day {day + 1}",
                "breakfast": daily_meal["Breakfast"],
                "lunch": daily_meal["Lunch"],
                "dinner": daily_meal["Dinner"],
                "snacks": daily_meal["Snacks"]
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
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

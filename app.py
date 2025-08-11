from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import threading
import time
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load model and scaler
model = joblib.load("model/XGBoost_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

# Define symptoms list globally
symptoms_list = [
    "fever", "sore_throat", "vomiting", "headache", "muscle_pain",
    "abdominal_pain", "diarrhea", "bleeding", "hearing_loss", "fatigue"
]

# Store latest sensor data
latest_sensor_data = {}
data_lock = threading.Lock()

@app.route("/", methods=["GET", "POST"])
def index():
    message = None
    message_class = None
    prediction_result = None

    if request.method == "POST":
        try:
            name = request.form.get("name")
            age = request.form.get("age")
            gender = request.form.get("gender")
            temperature = request.form.get("temperature")
            heart_rate = request.form.get("heart_rate")
            oxygen_level = request.form.get("oxygen_level")

            # Check if any required fields are missing
            if not all([name, age, gender, temperature, heart_rate, oxygen_level]) or \
               any(request.form.get(symptom) == '' for symptom in symptoms_list):
                message = "⚠️ Please fill in all fields to get a prediction"
                message_class = "danger"
            else:
                # Convert values
                age = int(age)
                temperature = float(temperature)
                heart_rate = float(heart_rate)
                oxygen_level = float(oxygen_level)
                symptoms = {k: int(request.form[k]) for k in symptoms_list}

                # Prepare input for model
                input_data = np.array([[*symptoms.values(), temperature, heart_rate, oxygen_level]])
                scaled_input = scaler.transform(input_data)
                prediction = model.predict(scaled_input)[0]
                prediction_result = "Positive" if prediction == 1 else "Negative"

                # Set display message
                if prediction_result == "Positive":
                    message = "⚠️ Positive for Lassa Fever! Please seek medical attention immediately."
                    message_class = "danger"
                else:
                    message = "✅ Negative for Lassa Fever. Stay safe and healthy!"
                    message_class = "success"

                # Save to Supabase
                data = {
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "prediction": prediction_result,
                    **symptoms,
                    "temperature": temperature,
                    "heart_rate": heart_rate,
                    "oxygen_level": oxygen_level
                }
                supabase.table("lassa_predictions").insert(data).execute()

        except Exception as e:
            app.logger.error(f"Error processing form: {str(e)}")
            message = "⚠️ Error processing your input. Please check your entries."
            message_class = "danger"

    return render_template("index.html", 
                           symptoms=symptoms_list,
                           message=message,
                           message_class=message_class)

# API for sensor to send vitals (POST JSON)
@app.route("/api/vitals", methods=["POST"])
def api_vitals():
    global latest_sensor_data
    data = request.json
    
    # Extract and store vital signs
    with data_lock:
        latest_sensor_data = {
            "temperature": data["temperature"],
            "heart_rate": data["heart_rate"],
            "oxygen_level": data["oxygen_level"],
            "timestamp": time.time()
        }
    
    # Process prediction (existing functionality)
    symptoms = {key: int(data[key]) for key in symptoms_list}
    temperature = float(data["temperature"])
    heart_rate = float(data["heart_rate"])
    oxygen_level = float(data["oxygen_level"])

    input_data = np.array([[*symptoms.values(), temperature, heart_rate, oxygen_level]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    prediction_result = "Positive" if prediction == 1 else "Negative"

    # Save to Supabase
    data["prediction"] = prediction_result
    supabase.table("lassa_predictions").insert(data).execute()

    return jsonify({"prediction": prediction_result})

# Endpoint to get latest sensor data
@app.route("/api/latest_vitals", methods=["GET"])
def get_latest_vitals():
    with data_lock:
        return jsonify(latest_sensor_data)

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

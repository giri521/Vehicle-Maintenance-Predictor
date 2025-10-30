from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib
import numpy as np
import requests
from datetime import datetime
import os # Added for better secret key handling

app = Flask(__name__)
# IMPORTANT: Use a complex, randomly generated secret key in production
# Using an environment variable is the best practice for deployment
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super_secret_key_default_fallback_use_real_one") 

# ------------------- BACKENDLESS CONFIG -------------------
# NOTE: These keys should ideally be loaded from environment variables for security.
BACKENDLESS_APP_ID = "665EA70C-F1D1-4EC9-919A-CD75C7A21363"  # 🔧 Replace with your real App ID
BACKENDLESS_API_KEY = "458F8A4C-8C26-410E-91B9-887CA0CFB808"  # 🔧 Replace with your REST API Key
USER_URL = f"https://api.backendless.com/{BACKENDLESS_APP_ID}/{BACKENDLESS_API_KEY}/data/vehicle_users"
HISTORY_URL = f"https://api.backendless.com/{BACKENDLESS_APP_ID}/{BACKENDLESS_API_KEY}/data/vehicle_history"
# Note: Ensure the 'vehicle_users' and 'vehicle_history' tables exist in your Backendless app.


# ------------------- MODEL LOADING -------------------
# Assuming 'model.pkl' and 'scaler.pkl' are available in the same directory
try:
    # NOTE: joblib.load requires model.pkl and scaler.pkl files to exist.
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Machine Learning Model and Scaler loaded successfully.")
except FileNotFoundError:
    print("⚠️ Warning: model.pkl or scaler.pkl not found. Prediction functionality will be disabled.")
    model = None
    scaler = None
except Exception as e:
    print(f"⚠️ Error loading model/scaler: {e}")
    model = None
    scaler = None


# ------------------- ROUTES -------------------

@app.route('/')
def main():
    """Login/Register page."""
    # Renders main.html (MUST BE CREATED)
    return render_template('main.html')

# ------------------- REGISTER -------------------
@app.route('/register', methods=['POST'])
def register():
    """Handles new user registration."""
    try:
        data = {
            "owner_name": request.form['owner_name'],
            "email": request.form['email'],
            "vehicle_number": request.form['vehicle_number'].upper().replace(" ", ""),
            "password": request.form['password']
        }

        # Check if vehicle already registered
        query_url = f"{USER_URL}?where=vehicle_number='{data['vehicle_number']}'"
        res = requests.get(query_url)
        
        # Check if request was successful AND records were returned
        if res.status_code == 200 and len(res.json()) > 0:
            return "⚠️ Vehicle already registered. Please login instead."

        # Save new user to Backendless
        res = requests.post(USER_URL, json=data)
        if res.status_code == 200:
            # Optionally log the user in immediately
            session['user'] = res.json()  
            return redirect(url_for('home'))
        else:
            return f"Registration failed: {res.text}"

    except Exception as e:
        return f"Error during registration: {str(e)}"

# ------------------- LOGIN -------------------
@app.route('/login', methods=['POST'])
def login():
    """Handles user login."""
    try:
        vehicle_number = request.form['vehicle_number'].upper().replace(" ", "")
        password = request.form['password']

        query_url = f"{USER_URL}?where=vehicle_number='{vehicle_number}'"
        res = requests.get(query_url)

        if res.status_code == 200:
            users = res.json()
            if len(users) == 0:
                return "❌ Vehicle not found. Please register first."

            user = users[0]
            # NOTE: For real-world security, you must use password hashing (e.g., bcrypt)
            if user['password'] == password:
                session['user'] = user
                return redirect(url_for('home'))
            else:
                return "⚠️ Incorrect password. Try again."
        else:
            return f"Login failed: {res.text}"

    except Exception as e:
        return f"Error during login: {str(e)}"

# ------------------- HOME (Prediction Page) -------------------
@app.route('/home')
def home():
    """Displays the prediction input page and the last prediction dashboard."""
    if 'user' not in session:
        return redirect(url_for('main'))
    
    user = session['user']
    recent_prediction = None
    
    # 1. Fetch the most recent prediction for this vehicle
    # Use 'sortBy=-created' to get the newest record first, and 'pageSize=1' for only one result
    query_url = (f"{HISTORY_URL}?where=vehicle_number='{user['vehicle_number']}'"
                 f"&sortBy=-created&pageSize=1")
    
    try:
        res = requests.get(query_url)
        if res.status_code == 200 and len(res.json()) > 0:
            recent_prediction = res.json()[0]
    except Exception as e:
        print(f"Error fetching recent prediction: {e}")
        # Continue even if fetching fails, just recent_prediction will be None

    # Renders index.html (MUST BE CREATED)
    return render_template('index.html', user=user, recent=recent_prediction)

# ------------------- PREDICT -------------------
@app.route('/predict', methods=['POST'])
def predict():
    """Processes input data, makes a prediction, saves history, and shows results."""
    if 'user' not in session:
        return redirect(url_for('main'))
        
    if not model or not scaler:
        return "ML model is not loaded. Cannot perform prediction."

    user = session['user']

    try:
        # Get data and convert to correct types
        data = {
            'Engine Temperature (°C)': float(request.form['engine_temp']),
            'Oil Pressure (bar)': float(request.form['oil_pressure']),
            'Vibration Level (Hz)': float(request.form['vibration']),
            'Battery Voltage (V)': float(request.form['battery_voltage']),
            'Mileage (km)': float(request.form['mileage']),
            'Fuel Efficiency (km/l)': float(request.form['fuel_efficiency'])
        }

        # --- ML Prediction ---
        features = np.array(list(data.values())).reshape(1, -1)
        scaled = scaler.transform(features)
        pred = model.predict(scaled)[0] # 0 for No Maintenance, 1 for Maintenance

        # --- Risk Calculation & Analysis (Heuristic based on common sense) ---
        section_risk = {
            # Risk increases as it deviates from an ideal temperature of 90°C
            "Engine": round(min(100, abs(data['Engine Temperature (°C)'] - 90) * 1.2), 2), 
            # Risk increases as it deviates from an ideal oil pressure of 3.5 bar
            "Oil Pressure": round(min(100, abs(data['Oil Pressure (bar)'] - 3.5) * 15), 2), 
            # Risk increases linearly with vibration
            "Vibration": round(min(100, data['Vibration Level (Hz)'] * 20), 2), 
            # Risk increases as it deviates from an ideal off-state voltage of 12.6V
            "Battery": round(min(100, abs(data['Battery Voltage (V)'] - 12.6) * 10), 2), 
            # Linear wear over 200,000 km
            "Mileage": round(min(100, (data['Mileage (km)'] / 200000) * 100), 2), 
            # Risk increases as it deviates from an ideal efficiency of 15 km/l
            "Fuel System": round(min(100, abs(data['Fuel Efficiency (km/l)'] - 15) * 5), 2)
        }

        overall_risk = round(sum(section_risk.values()) / len(section_risk), 2)

        # Detailed analysis and service recommendations
        reasons = {
            "Engine": "High temperature suggests potential cooling or oil issues. Ideal operating range is 85-105°C.",
            "Oil Pressure": "Deviation from 3.0-4.0 bar can cause severe damage to internal components.",
            "Vibration": "High vibration (ideally < 1 Hz) suggests issues with mounts, tires, or engine balance.",
            "Battery": "Voltage outside the 12.4V-12.8V (off) range indicates charging or capacity issues.",
            "Mileage": "High mileage triggers the need for comprehensive inspection of wear items like belts, bushings, and brakes.",
            "Fuel System": "Poor efficiency suggests fuel filter clogs, injector issues, or sensor malfunctions."
        }

        services = {
            "Engine": "Check coolant level, thermostat, water pump, and radiator for blockages.",
            "Oil Pressure": "Inspect oil level, change oil/filter, and check the oil pump's integrity.",
            "Vibration": "Inspect engine mounts, balance tires, and check wheel alignment.",
            "Battery": "Clean terminals, check alternator output, and perform a full load test.",
            "Mileage": "Perform a major service including timing belt inspection and fluid changes.",
            "Fuel System": "Replace the fuel filter, clean injectors, and inspect air filter integrity."
        }

        result = "🟢 No Immediate Maintenance Required"
        if pred == 1 or overall_risk > 50: # Trigger maintenance if ML predicts it OR overall risk is high
            result = "🔴 Maintenance Required Soon!"
        if overall_risk > 75: # Higher threshold for URGENT message
             result = "🔥 URGENT Maintenance Required!"


        # --- Save to Backendless History ---
        history_data = {
            "vehicle_number": user['vehicle_number'],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "ownerId": user.get('ownerId', 'anonymous'), # Link to the user object, use get for safety
            "overall_risk": overall_risk,
            "result": result,
            # Flatten feature data for easy storage and retrieval
            "engine_temp": data['Engine Temperature (°C)'],
            "oil_pressure": data['Oil Pressure (bar)'],
            "vibration": data['Vibration Level (Hz)'],
            "battery_voltage": data['Battery Voltage (V)'],
            "mileage": data['Mileage (km)'],
            "fuel_efficiency": data['Fuel Efficiency (km/l)']
        }
        
        # Save the full prediction data to the history table
        requests.post(HISTORY_URL, json=history_data) 

        # --- Render Result Page ---
        # Renders result.html (MUST BE CREATED)
        return render_template(
            'result.html',
            result=result,
            data=data,
            section_risk=section_risk,
            reasons=reasons,
            services=services,
            overall_risk=overall_risk
        )

    except ValueError:
        return "Error: Invalid input. All fields must be numeric.", 400
    except Exception as e:
        # Catch exceptions like issues with requests/Backendless connection
        return f"A server error occurred during prediction: {str(e)}", 500

# ------------------- HISTORY -------------------
@app.route('/history')
def history():
    """Displays the full prediction history for the logged-in vehicle."""
    if 'user' not in session:
        return redirect(url_for('main'))

    user = session['user']
    history_data = []

    # Fetch all history records for this vehicle, sorted by newest first
    query_url = (f"{HISTORY_URL}?where=vehicle_number='{user['vehicle_number']}'"
                 f"&sortBy=-timestamp") # Sort by timestamp descending

    try:
        res = requests.get(query_url)
        if res.status_code == 200:
            history_data = res.json()
    except Exception as e:
        print(f"Error fetching history: {e}")
        # history_data remains an empty list

    # Re-map keys to match history.html template expectations
    for record in history_data:
        # Use .get for robustness in case a field is missing
        record['Engine Temperature (°C)'] = record.get('engine_temp')
        record['Oil Pressure (bar)'] = record.get('oil_pressure')
        record['Vibration Level (Hz)'] = record.get('vibration')
        record['Battery Voltage (V)'] = record.get('battery_voltage')
        record['Mileage (km)'] = record.get('mileage')
        record['Fuel Efficiency (km/l)'] = record.get('fuel_efficiency')

    # Renders history.html (MUST BE CREATED)
    return render_template('history.html', user=user, history_data=history_data)

# ------------------- LOGOUT -------------------
@app.route('/logout')
def logout():
    """Clears the session and redirects to the main login/register page."""
    session.clear()
    return redirect(url_for('main'))

if __name__ == '__main__':
    # Set host='0.0.0.0' for deployment
    app.run(debug=True)

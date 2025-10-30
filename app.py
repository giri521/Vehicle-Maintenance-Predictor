from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib
import numpy as np
import requests
from datetime import datetime

app = Flask(__name__)
# IMPORTANT: Use a complex, randomly generated secret key in production
app.secret_key = "super_secret_key" 

# ------------------- BACKENDLESS CONFIG -------------------
BACKENDLESS_APP_ID = "665EA70C-F1D1-4EC9-919A-CD75C7A21363"  # 🔧 Replace with your real App ID
BACKENDLESS_API_KEY = "458F8A4C-8C26-410E-91B9-887CA0CFB808"  # 🔧 Replace with your REST API Key
USER_URL = f"https://api.backendless.com/{BACKENDLESS_APP_ID}/{BACKENDLESS_API_KEY}/data/vehicle_users"
HISTORY_URL = f"https://api.backendless.com/{BACKENDLESS_APP_ID}/{BACKENDLESS_API_KEY}/data/vehicle_history"
# Note: Ensure the 'vehicle_users' and 'vehicle_history' tables exist in your Backendless app.


# ------------------- MODEL LOADING -------------------
# Assuming 'model.pkl' and 'scaler.pkl' are available in the same directory
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Machine Learning Model and Scaler loaded successfully.")
except FileNotFoundError:
    print("⚠️ Warning: model.pkl or scaler.pkl not found. Prediction functionality will fail.")
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
    query_url = (f"{HISTORY_URL}?where=vehicle_number='{user['vehicle_number']}'"
                  f"&sortBy=-created&pageSize=1")
    
    try:
        res = requests.get(query_url)
        if res.status_code == 200 and len(res.json()) > 0:
            recent_prediction = res.json()[0]
    except Exception as e:
        print(f"Error fetching recent prediction: {e}")

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

        # --- Risk Calculation & Analysis ---
        section_risk = {
            "Engine": min(100, abs(data['Engine Temperature (°C)'] - 90) * 1.2),
            "Oil Pressure": min(100, abs(data['Oil Pressure (bar)'] - 3.5) * 15),
            "Vibration": min(100, data['Vibration Level (Hz)'] * 2),
            "Battery": min(100, abs(data['Battery Voltage (V)'] - 12.6) * 10),
            "Mileage": min(100, (data['Mileage (km)'] / 200000) * 100),
            "Fuel System": min(100, abs(data['Fuel Efficiency (km/l)'] - 15) * 5)
        }

        overall_risk = round(sum(section_risk.values()) / len(section_risk), 2)

        # --- New Status Logic based on Risk and ML Prediction ---
        maintenance_status = "🟢 Good" # Default for 0-15%

        if overall_risk > 65:
            # 66 - 100: Immediate Required
            maintenance_status = "🔥 IMMEDIATE Required!"
        elif overall_risk > 30:
            # 31 - 65: Try Immediate Required
            maintenance_status = "🔴 Try Immediate Required"
        elif overall_risk > 15:
            # 16 - 30: Required Soon
            maintenance_status = "🟡 Required Soon"
        
        # If ML predicts maintenance, ensure the status reflects a need for service
        if pred == 1:
            if overall_risk <= 15: 
                 maintenance_status = "🟡 Required Soon"
            elif overall_risk > 65:
                 maintenance_status = "🔥 IMMEDIATE Required!"
            elif overall_risk > 30:
                 maintenance_status = "🔴 Try Immediate Required"
            else: # 16-30%
                 maintenance_status = "🟡 Required Soon"
        
        # Determine the simple result message (used on result.html for older logic)
        if "IMMEDIATE" in maintenance_status:
            result = "🔥 URGENT Maintenance Required!"
        elif "Required" in maintenance_status:
            result = "🔴 Maintenance Required Soon!"
        else:
            result = "🟢 No Immediate Maintenance Required"
            

        reasons = {
            "Engine": "High temperature may cause wear and tear or coolant issues. Ideal range is 85-105°C.",
            "Oil Pressure": "Low or high oil pressure can damage moving parts. Ideal range is 3.0-4.0 bar.",
            "Vibration": "Excess vibration suggests imbalance, worn mounts, or engine issue. Ideal is below 1 Hz.",
            "Battery": "Voltage instability can affect electrical systems. Ideal is 12.4V-12.8V (off) or 13.5V-14.5V (on).",
            "Mileage": "Higher mileage indicates natural wear of major components.",
            "Fuel System": "Poor efficiency suggests fuel filter, injector, or air intake issues."
        }

        services = {
            "Engine": "Check coolant level, thermostat, water pump, and oil quality.",
            "Oil Pressure": "Check oil level, oil filter condition, and oil pump integrity.",
            "Vibration": "Inspect engine mounts, check wheel/tire balance, or review suspension.",
            "Battery": "Check battery terminals, alternator, and run a load test.",
            "Mileage": "Perform a major service, inspect brake pads and belts.",
            "Fuel System": "Clean fuel injectors, replace fuel filter, inspect air filter."
        }


        # --- Save to Backendless History ---
        history_data = {
            "vehicle_number": user['vehicle_number'],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "ownerId": user['ownerId'], 
            "overall_risk": overall_risk,
            "result": result, 
            "maintenance_status": maintenance_status, # NEW field for detailed status
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
        return render_template(
            'result.html',
            result=result,
            data=data,
            section_risk=section_risk,
            reasons=reasons,
            services=services,
            overall_risk=overall_risk,
            maintenance_status=maintenance_status 
        )

    except ValueError:
        return "Error: Invalid input. All fields must be numeric.", 400
    except Exception as e:
        return f"Error: {str(e)}", 500

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

    # Re-map keys and ensure data integrity for the new 'maintenance_status' field
    for record in history_data:
        # Safely retrieve maintenance_status. Use .get() to handle missing keys from old records.
        status = record.get('maintenance_status')

        # FIX: If the status is missing or None, calculate a default status based on risk
        if status is None:
             # Default risk to 0 in case the key is missing entirely (highly unlikely here, but safe)
             risk = record.get('overall_risk', 0) 
             
             # Re-apply the logic used in /predict to create the missing status string
             if risk > 65:
                 record['maintenance_status'] = "🔥 IMMEDIATE Required!"
             elif risk > 30:
                 record['maintenance_status'] = "🔴 Try Immediate Required"
             elif risk > 15:
                 record['maintenance_status'] = "🟡 Required Soon"
             else:
                 record['maintenance_status'] = "🟢 Good"


        # Re-map feature data for the template
        record['Engine Temperature (°C)'] = record.pop('engine_temp')
        record['Oil Pressure (bar)'] = record.pop('oil_pressure')
        record['Vibration Level (Hz)'] = record.pop('vibration')
        record['Battery Voltage (V)'] = record.pop('battery_voltage')
        record['Mileage (km)'] = record.pop('mileage')
        record['Fuel Efficiency (km/l)'] = record.pop('fuel_efficiency')

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

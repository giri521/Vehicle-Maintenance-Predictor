from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        data = {
            'Engine Temperature (°C)': float(request.form['engine_temp']),
            'Oil Pressure (bar)': float(request.form['oil_pressure']),
            'Vibration Level (Hz)': float(request.form['vibration']),
            'Battery Voltage (V)': float(request.form['battery_voltage']),
            'Mileage (km)': float(request.form['mileage']),
            'Fuel Efficiency (km/l)': float(request.form['fuel_efficiency'])
        }

        # Prepare input
        features = np.array(list(data.values())).reshape(1, -1)
        scaled = scaler.transform(features)

        # Predict maintenance
        pred = model.predict(scaled)[0]

        # Calculate section-wise risk (mock logic for visualization)
        section_risk = {
            "Engine": min(100, abs(data['Engine Temperature (°C)'] - 90) * 1.2),
            "Oil Pressure": min(100, abs(data['Oil Pressure (bar)'] - 3) * 15),
            "Vibration": min(100, data['Vibration Level (Hz)'] * 2),
            "Battery": min(100, abs(data['Battery Voltage (V)'] - 12.5) * 10),
            "Mileage": min(100, (data['Mileage (km)'] / 200000) * 100),
            "Fuel System": min(100, abs(data['Fuel Efficiency (km/l)'] - 15) * 5)
        }

        overall_risk = round(sum(section_risk.values()) / len(section_risk), 2)

        # Define reasons
        reasons = {
            "Engine": "High temperature may cause wear and tear.",
            "Oil Pressure": "Low oil pressure can damage moving parts.",
            "Vibration": "Excess vibration suggests imbalance or engine issue.",
            "Battery": "Voltage instability can affect electrical systems.",
            "Mileage": "Higher mileage indicates natural wear of components.",
            "Fuel System": "Poor efficiency suggests fuel filter or injector issues."
        }

        # Suggested maintenance services
        services = {
            "Engine": "Oil change, coolant check, spark plug inspection",
            "Oil Pressure": "Oil filter replacement, pump inspection",
            "Vibration": "Engine mount check, wheel alignment",
            "Battery": "Terminal cleaning, voltage check",
            "Mileage": "General inspection, replace worn-out parts",
            "Fuel System": "Injector cleaning, fuel filter check"
        }

        # Maintenance message
        result = "🟢 No Immediate Maintenance Required"
        if pred == 1 or overall_risk > 60:
            result = "🔴 Maintenance Required Soon!"

        return render_template(
            'result.html',
            result=result,
            data=data,
            section_risk=section_risk,
            reasons=reasons,
            services=services,
            overall_risk=overall_risk
        )

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

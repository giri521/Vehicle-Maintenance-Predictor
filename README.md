# Vehicle Maintenance Predictor

[Deployed Link](https://vehicle-maintenance-predictor.onrender.com/)

## Overview

Vehicle Maintenance Predictor is a web-based application that predicts vehicle maintenance requirements using machine learning models based on sensor data. It is designed to help vehicle owners and fleet managers anticipate potential issues and schedule preventive maintenance, reducing downtime and maintenance costs.

## Features

* Predict maintenance requirements based on real-time sensor data.
* Input parameters include engine temperature, oil pressure, vibration, and battery voltage.
* Web interface for easy interaction.
* Machine learning model trained on historical vehicle sensor data.

## Technologies Used

* Python
* Flask
* HTML/CSS
* Machine Learning (Scikit-learn)
* Pandas, NumPy

## Getting Started

### Prerequisites

* Python 3.x
* pip

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/giri521/Vehicle-Maintenance-Predictor.git
   ```
2. Navigate to the project directory:

   ```
   cd Vehicle-Maintenance-Predictor
   ```
3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```
4. Run the Flask app:

   ```
   python app.py
   ```
5. Open your browser and visit [http://127.0.0.1:5000](http://127.0.0.1:5000) or use the deployed link.

## Usage

1. Open the web app.
2. Enter vehicle sensor data (Engine Temperature, Oil Pressure, Vibration, Battery Voltage).
3. Submit to get maintenance prediction.

## File Structure

* `app.py` - Flask web application.
* `train.py` - Script for training the machine learning model.
* `model.pkl` - Saved machine learning model.
* `scaler.pkl` - Saved scaler for preprocessing input data.
* `vehicle_sensor_data.csv` - Sample dataset for training the model.
* `templates/` - HTML templates for web interface.
* `requirements.txt` - Python dependencies.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.

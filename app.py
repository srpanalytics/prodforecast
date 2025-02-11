from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset
df = pd.read_csv("Train.csv")  # Update with the correct file path

# Define features and targets
X = df[["Sum of Quantity"]]
y_consumption = df["Sum of Consumption"]
y_byproduct = df["Sum of By product"]

# Train models
model_c = LinearRegression()
model_b = LinearRegression()
model_c.fit(X, y_consumption)
model_b.fit(X, y_byproduct)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        production_value = float(data['production'])
        user_production_array = np.array([[production_value]])

        predicted_consumption = model_c.predict(user_production_array)[0]
        predicted_byproduct = model_b.predict(user_production_array)[0]

        return jsonify({
            "consumption": round(predicted_consumption, 2),
            "byproduct": round(predicted_byproduct, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

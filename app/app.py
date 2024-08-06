#----------------------------------------------------------------------------------------
from flask import Flask, render_template, request, Markup
import numpy as np
import pickle

# Load the crop recommendation model
crop_recommendation_model_path = 'models/model_rf.pkl'

try:
    crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")

# Mapping from predicted index to crop name
crop_mapping = {
    1: 'rice',
    2: 'maize',
    3: 'chickpea',
    4: 'kidneybeans',
    5: 'pigeonpeas',
    6: 'mothbeans',
    7: 'mungbean',
    8: 'blackgram',
    9: 'lentil',
    10: 'pomegranate',
    11: 'banana',
    12: 'mango',
    13: 'grapes',
    14: 'watermelon',
    15: 'muskmelon',
    16: 'apple',
    17: 'orange',
    18: 'papaya',
    19: 'coconut',
    20: 'cotton',
    21: 'jute',
    22: 'coffee'
}

app = Flask(__name__)

# Home page
@app.route('/')
def home():
    title = 'KisanBandhu - Home'
    return render_template('index.html', title=title)

# Crop recommendation form page
@app.route('/crop')
def crop_recommend():
    title = 'KisanBandhu - Crop Recommendation'
    return render_template('crop.html', title=title)

# Crop recommendation result page
@app.route('/crop-prediction', methods=['POST'])
def crop_prediction():
    title = 'KisanBandhu - Crop Recommendation'

    if request.method == 'POST':
        try:
            # Debug: Check form fields
            print("Form data received:")
            for key in request.form:
                print(f"{key}: {request.form[key]}")

            N = int(request.form['nitrogen'])
            P = int(request.form['phosphorous'])
            K = int(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # Debug: Check data being passed to model
            print(f"Data for prediction: N={N}, P={P}, K={K}, Temp={temperature}, Humidity={humidity}, pH={ph}, Rainfall={rainfall}")

            # Prepare the data for prediction
            data = np.array([[N, P, K, temperature, humidity, ph]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            # Debug: Check prediction result
            print(f"Prediction result (index): {final_prediction}")

            # Map the prediction to the crop name
            predicted_crop_name = crop_mapping.get(final_prediction, "Unknown Crop")

            # Debug: Check mapped crop name
            print(f"Predicted Crop: {predicted_crop_name}")

            return render_template('crop-res.html', prediction=predicted_crop_name, title=title)

        except Exception as e:
            print(f"Error occurred during prediction: {e}")
            return render_template('try-again.html', message="An error occurred during prediction. Please try again.", title="Error")

if __name__ == '__main__':
    app.run(debug=True)



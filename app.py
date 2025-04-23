from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
with open('house_price_model.pkl', 'rb') as file:
    model = joblib.load(file)

print("Model type:", type(model))
print("Expected number of features:", model.n_features_in_)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data (Boston housing features)
    crim = float(request.form['crim'])
    zn = float(request.form['zn'])
    indus = float(request.form['indus'])
    chas = float(request.form['chas'])
    nox = float(request.form['nox'])
    rm = float(request.form['rm'])
    age = float(request.form['age'])
    dis = float(request.form['dis'])
    rad = float(request.form['rad'])
    tax = float(request.form['tax'])
    ptratio = float(request.form['ptratio'])
    b = float(request.form['b'])
    lstat = float(request.form['lstat'])
    
    # Compute RM_TAX_RATIO
    rm_tax_ratio = rm / tax if tax != 0 else 0  # Avoid division by zero
    
    # Create features array (14 features)
    features = [
        crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat, rm_tax_ratio
    ]
    
    # Convert to numpy array and reshape for prediction
    features_array = np.array(features).reshape(1, -1)
    print("Input features shape:", features_array.shape)  # Debug: Should be (1, 14)
    
    # Make prediction
    prediction = model.predict(features_array)[0]
    
    return render_template('result.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run()
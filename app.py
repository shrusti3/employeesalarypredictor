# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load('salary_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define options for dropdowns
dropdown_options = {
    'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
    'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 
                 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', 
                 '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
    'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 
                      'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
    'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 
                  'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 
                  'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 
                  'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
    'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
    'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
    'gender': ['Female', 'Male'],
    'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 
                      'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 
                      'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 
                      'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 
                      'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 
                      'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 
                      'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 
                      'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
}

@app.route('/')
def home():
    return render_template('index.html', options=dropdown_options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([data])
        
        # Convert numerical fields
        input_df['age'] = input_df['age'].astype(int)
        input_df['educational-num'] = input_df['educational-num'].astype(int)
        input_df['capital-gain'] = input_df['capital-gain'].astype(int)
        input_df['capital-loss'] = input_df['capital-loss'].astype(int)
        input_df['hours-per-week'] = input_df['hours-per-week'].astype(int)
        
        # Encode categorical variables
        for col in dropdown_options.keys():
            if col in input_df.columns:
                le = label_encoders[col]
                input_df[col] = le.transform(input_df[col])
        
        # Make sure we have all expected columns in the right order
        expected_columns = model.feature_names_in_.tolist()


        
        # Reorder columns and drop any extras
        input_df = input_df[expected_columns]
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Decode prediction
        income_pred = label_encoders['income'].inverse_transform(prediction)[0]
        
        return render_template('result.html', prediction=income_pred)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
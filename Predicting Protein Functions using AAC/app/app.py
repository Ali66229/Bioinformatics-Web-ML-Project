import os
from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

app = Flask(__name__)

# Load pre-trained model and label encoder
model = RandomForestClassifier(random_state=42, n_estimators=100)
label_encoder = LabelEncoder()



# Sample prediction function (replace with actual model prediction)
def make_prediction(model, label_encoder, data):

    return ['GO:0008601'] * len(data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the file into a DataFrame
    try:
        data = pd.read_csv(file)
        predictions = make_prediction(model, label_encoder, data)
        
        # Create a list of dictionaries for the output table
        prediction_results = [{'protein_id': protein_id, 'predicted_go_term': prediction} 
                              for protein_id, prediction in zip(data['Protein_ID'], predictions)]
        
        return jsonify(prediction_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

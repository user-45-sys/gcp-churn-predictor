from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running successfully!', 'version': '1.0'})
# Load the trained model that you generated in Phase 2
model = joblib.load('churn_model.pkl')

@app.route('/predict', methods=['POST'])
def predict_churn():
    try:
        # Get the JSON data sent by Salesforce
        data = request.get_json()
        
        # Convert the JSON into a Pandas DataFrame so the model can read it
        # Expected fields: Monthly_Spend, Tickets_Opened, Days_Since_Last_Login, Feature_Usage_Score
        df = pd.DataFrame([data])
        
        # Make the prediction (1 = Churned, 0 = Safe)
        prediction = model.predict(df)[0]
        
        # Get the probability (e.g., 0.85 means 85% chance of churning)
        probability = model.predict_proba(df)[0][1]
        
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'churn_probability_score': round(probability * 100, 2)
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    # Cloud Run requires the app to listen on the port defined by the PORT environment variable, defaulting to 8080
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

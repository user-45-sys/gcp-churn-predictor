import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load the synthetic customer data
print("Loading customer data...")
df = pd.read_csv('customer_data.csv')

# 2. Define the 'Features' (the behavior we analyze) and the 'Target' (what we want to predict)
features = ['Monthly_Spend', 'Tickets_Opened', 'Days_Since_Last_Login', 'Feature_Usage_Score']
X = df[features]
y = df['Churned']

# 3. Split the data: 80% to train the AI, 20% to test its accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the Random Forest Classifier
print("Training the predictive model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Calculate and print the model's accuracy
accuracy = model.score(X_test, y_test)
print(f"Model trained successfully! Accuracy: {accuracy * 100:.2f}%")

# 6. Export the trained model as a file so we can deploy it to Google Cloud
joblib.dump(model, 'churn_model.pkl')
print("Model exported successfully as 'churn_model.pkl'")
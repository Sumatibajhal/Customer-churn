import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, render_template

# Step 1: Load the dataset
data = pd.read_csv('D:/Code Playground/Python course/customer churn/Churn_Modelling.csv')  # Replace with the actual path to your dataset

# Step 2: Preprocess the data
# Drop irrelevant columns
target_col = 'Exited'
X = data.drop(['RowNumber', 'CustomerId', 'Surname', target_col], axis=1)
y = data[target_col]

# Separate numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns
categorical_cols = X.select_dtypes(exclude=[np.number]).columns

# Handle missing values
imputer_numeric = SimpleImputer(strategy='mean')
X[numeric_cols] = imputer_numeric.fit_transform(X[numeric_cols])

imputer_categorical = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = imputer_categorical.fit_transform(X[categorical_cols])

# Encode categorical variables
encoder = LabelEncoder()
for col in categorical_cols:
    X[col] = encoder.fit_transform(X[col])

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(feature_importances)

# Step 6: Develop a UI using Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Process form data
        user_input = [float(request.form[field]) for field in X.columns]
        prediction = model.predict([user_input])
        return f"The predicted churn status is: {'Churn' if prediction[0] == 1 else 'No Churn'}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

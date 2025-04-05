import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

model = joblib.load('./ML_Models/diab.pkl')

df = pd.read_csv('./Trained_Data/diabetes_prediction_dataset.csv')

# Drop the columns that are not used in the model
columns_to_drop = ['HbA1c_level', 'blood_glucose_level']
df = df.drop(columns=columns_to_drop)

if df.select_dtypes(include=['object']).shape[1] > 0:
    df = pd.get_dummies(df, drop_first=True)

scaler = StandardScaler()
X = df.drop(columns=['diabetes'])
feature_names = X.columns.tolist()
scaler.fit(X)

def Diabetes(user_input):
    try:
        print("\n=== Diabetes Prediction Debug Info ===")
        print("Expected features:", feature_names)
        print("Number of expected features:", len(feature_names))
        print("\nInput features:", list(user_input.keys()))
        print("Number of input features:", len(user_input))
        
        # Create a dictionary with all features initialized to 0
        input_dict = {feature: 0 for feature in feature_names}
        
        # Update with provided values
        for key, value in user_input.items():
            if key in input_dict:
                input_dict[key] = value
                print(f"Matched feature: {key} = {value}")
            else:
                print(f"Warning: Feature {key} not found in expected features")

        # Check for missing required features
        missing_features = [feature for feature in feature_names if feature not in user_input]
        if missing_features:
            print(f"\nMissing required features: {missing_features}")
            return f"Error: Missing required features: {', '.join(missing_features)}"

        # Convert to array in the correct order
        input_array = np.array([input_dict[feature] for feature in feature_names]).reshape(1, -1)
        print("\nInput array shape:", input_array.shape)
        
        # Print the actual values being sent to the model
        print("\nValues being sent to model:")
        for i, (feature, value) in enumerate(zip(feature_names, input_array[0])):
            print(f"{feature}: {value}")
        
        input_scaled = scaler.transform(input_array)
        prediction = model.predict_proba(input_scaled)[0][1]  # Get probability of being diabetic
        print("=== End Debug Info ===\n")
        return float(prediction)  # Return probability as a float
    
    except Exception as e:
        print(f"\nError details: {str(e)}")
        return f"Error in prediction: {str(e)}"

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Hypertension Model
hypertension_model = joblib.load('./ML_Models/hyp.pkl')

hypertension_df = pd.read_csv('./Trained_Data/hypertension_data.csv')

# Drop the columns that are not used in the model
columns_to_drop = ['fbs', 'exang', 'chol', 'trestbps', 'oldpeak', 'ca', 'thal']
hypertension_df = hypertension_df.drop(columns=columns_to_drop)

if hypertension_df.select_dtypes(include=['object']).shape[1] > 0:
    hypertension_df = pd.get_dummies(hypertension_df, drop_first=True)

hypertension_scaler = StandardScaler()
X_hypertension = hypertension_df.drop(columns=['target'])
feature_names = X_hypertension.columns.tolist()
hypertension_scaler.fit(X_hypertension)

def Hypertension(user_input):
    try:
        print("\n=== Hypertension Prediction Debug Info ===")
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
        
        # Ensure we have exactly 6 features
        if input_array.shape[1] != 6:
            print(f"\nWarning: Input array has {input_array.shape[1]} features but model expects 6")
            # Take only the first 6 features
            input_array = input_array[:, :6]
            print("Adjusted input array shape:", input_array.shape)
        
        input_scaled = hypertension_scaler.transform(input_array)
        
        # Use decision_function for SVC model
        decision_score = hypertension_model.decision_function(input_scaled)[0]
        # Convert decision score to probability using sigmoid function
        probability = 1 / (1 + np.exp(-decision_score))
        print("=== End Debug Info ===\n")
        return float(probability)  # Return probability as a float
    
    except Exception as e:
        print(f"\nError details: {str(e)}")
        return f"Error in prediction: {str(e)}"
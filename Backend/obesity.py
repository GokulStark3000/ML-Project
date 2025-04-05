import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import traceback

try:
    # Load Obesity Model
    print("Loading obesity model...")
    obesity_model = joblib.load('./ML_Models/obs.pkl')
    print("Model loaded successfully")

    # Read dataset
    print("\nReading dataset...")
    obesity_df = pd.read_csv('./Trained_Data/obesity_level.csv')
    print("Dataset loaded successfully")

    # Standardize column names
    obesity_df.columns = obesity_df.columns.str.strip()

    # Drop unnecessary columns
    columns_to_drop = ['FAVC', 'FCVC', 'NCP', 'SMOKE', 'SCC', 'FAF', 'TUE', 'MTRANS', 'id']
    existing_columns_to_drop = [col for col in columns_to_drop if col in obesity_df.columns]
    obesity_df = obesity_df.drop(columns=existing_columns_to_drop, errors='ignore')

    # Check if target column exists
    if '0be1dad' in obesity_df.columns:
        X_obesity = obesity_df.drop(columns=['0be1dad'])
    else:
        raise KeyError("Target column '0be1dad' not found")

    # Convert categorical variables to dummy variables
    categorical_columns = X_obesity.select_dtypes(include=['object']).columns
    X_obesity = pd.get_dummies(X_obesity, columns=categorical_columns, drop_first=True)

    # Get the model's expected feature count
    model_feature_count = obesity_model.n_features_in_
    print(f"\nModel expects {model_feature_count} features")

    # Select only the first 8 features to match the model's expectations
    feature_names = X_obesity.columns.tolist()[:model_feature_count]
    X_obesity = X_obesity[feature_names]

    # Prepare the scaler
    obesity_scaler = StandardScaler()
    obesity_scaler.fit(X_obesity)
    print("Scaler fitted successfully")

except Exception as e:
    print(f"Error during initialization: {str(e)}")
    print("Traceback:")
    print(traceback.format_exc())
    raise

def Obesity(user_input):
    try:
        # Input validation
        if 'Height' in user_input:
            height = user_input['Height']
            if height < 1.45 or height > 1.97:
                return f"Error: Height must be between 1.45m and 1.97m. Provided: {height}m"
        
        if 'Weight' in user_input:
            weight = user_input['Weight']
            if weight < 39 or weight > 165:
                return f"Error: Weight must be between 39kg and 165kg. Provided: {weight}kg"
        
        if 'Age' in user_input:
            age = user_input['Age']
            if age < 14 or age > 61:
                return f"Error: Age must be between 14 and 61 years. Provided: {age} years"
        
        if 'CH2O' in user_input:
            ch2o = user_input['CH2O']
            if ch2o < 1.0 or ch2o > 3.0:
                return f"Error: CH2O must be between 1.0 and 3.0. Provided: {ch2o}"
        
        # Calculate BMI for primary classification
        if 'Height' in user_input and 'Weight' in user_input:
            height = user_input['Height']
            weight = user_input['Weight']
            bmi = weight / (height * height)
            
            # BMI-based classification
            if bmi < 18.5:
                bmi_category = "Insufficient_Weight"
            elif bmi >= 18.5 and bmi < 25:
                bmi_category = "Normal_Weight"
            elif bmi >= 25 and bmi < 30:
                bmi_category = "Overweight_Level_I"
            elif bmi >= 30 and bmi < 35:
                bmi_category = "Obesity_Type_I"
            elif bmi >= 35 and bmi < 40:
                bmi_category = "Obesity_Type_II"
            else:
                bmi_category = "Obesity_Type_III"
        
        # Create a dictionary with all features initialized to 0
        input_dict = {feature: 0 for feature in feature_names}
        
        # Handle categorical variables
        if 'Gender' in user_input:
            if user_input['Gender'] == 1:
                input_dict['Gender_Male'] = 1
                
        if 'CAEC' in user_input:
            caec_value = user_input['CAEC']
            if caec_value == 1:
                input_dict['CAEC_Always'] = 1
            elif caec_value == 2:
                input_dict['CAEC_Frequently'] = 1
            elif caec_value == 3:
                input_dict['CAEC_Sometimes'] = 1
                
        if 'CALC' in user_input:
            calc_value = user_input['CALC']
            if calc_value == 1:
                input_dict['CALC_Frequently'] = 1
            elif calc_value == 2:
                input_dict['CALC_Sometimes'] = 1
        
        # Update with provided values for numerical features
        for key, value in user_input.items():
            if key in ['Age', 'Height', 'Weight', 'CH2O']:
                input_dict[key] = value
            elif key not in ['Gender', 'CAEC', 'CALC', 'id']:
                input_dict[key] = value

        # Check for missing required features
        missing_features = [feature for feature in feature_names if feature not in input_dict]
        if missing_features:
            return f"Error: Missing required features: {', '.join(missing_features)}"

        # Convert to array in correct order
        input_array = np.array([input_dict[feature] for feature in feature_names]).reshape(1, -1)
        input_scaled = obesity_scaler.transform(input_array)
        
        # Get prediction and probabilities
        prediction = obesity_model.predict(input_scaled)[0]
        
        # Map numerical predictions to descriptive labels
        obesity_levels = {
            0: "Insufficient_Weight",
            1: "Normal_Weight",
            2: "Overweight_Level_I",
            3: "Overweight_Level_II",
            4: "Obesity_Type_I",
            5: "Obesity_Type_II",
            6: "Obesity_Type_III"
        }
        
        model_prediction = obesity_levels.get(int(prediction), "Unknown")
        
        # Use BMI-based prediction if it differs significantly from model prediction
        if bmi_category != model_prediction:
            return bmi_category
        
        return model_prediction
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return f"Error in prediction: {str(e)}"

import json
import joblib
import numpy as np
import ast
from sklearn.model_selection import train_test_split
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

import seaborn as sns
import os
import sys


def process_advisor_request(request, model_path, feature_names, X_train, mode):
    """
    Process the advisor request, create an input vector, and return predictions.
    """
    # Extract data from the request
    class_data = request['data']
    target_class = request['option']

    # Initialize the feature vector with default values (-1 for missing classes)
    input_row = {col: -1 for col in feature_names}

    # Populate the feature vector with grades from the request
    for item in class_data:
        class_name = item['class']
        grade = float(item['grade'])
        if class_name in feature_names:
            input_row[class_name] = int(grade)

    # Convert the input row to a DataFrame
    X_new = pd.DataFrame([input_row])

    # Load the model dictionary and extract the model
    model_dict = joblib.load(model_path)
    model = model_dict['model']

    # Predict using the model
    prediction = model.predict(X_new)[0]

    print("Prediction result:", prediction)

    # Class conversion mapping
    if mode == 'grade':
        class_converting = {
            0: 'A+/A/A-/S',
            1: 'B+/B/B-',
            2: 'C+/C/C-',
            3: 'Fail'
        }
    else:
        class_converting = {
            0: 'FAIL',
            1: 'PASS'
        }

    # Get feature importances if available
#    if hasattr(model, 'feature_importances_'):
#        feature_importances = model.feature_importances_
#        importance_df = pd.DataFrame({
#            "Feature": feature_names,
#            "Importance": feature_importances
#        }).sort_values(by="Importance", ascending=False)
#
#        # Save feature importance plot
#        feature_importance_path = f"./images/feature_importance_{target_class}.png"
#        plt.figure(figsize=(10, 8))
#        sns.barplot(x="Importance", y="Feature", data=importance_df.head(20))
#        plt.title(f"Top 20 Feature Importances for {target_class}")
#        plt.xlabel("Importance")
#        plt.ylabel("Feature")
#        plt.tight_layout()
#        plt.savefig(feature_importance_path)
#        plt.close()
#
#        # Format the output
#        output = {
#            "output": [
#                {"type": "text", "content": f"Predicted grade for {target_class} is: {class_converting[prediction]}"},
#                {"type": "text", "content": f"Top 20 Features by Importance:\n{importance_df.head(20)}"},
#                {"type": "image", "content": feature_importance_path}
#            ]
#        }
#    else:
        output = {
            "output": [
                {"type": "text", "content": f"Predicted grade for {target_class} is: {class_converting[prediction]}"},
                {"type": "text", "content": "Feature importances are not available for this model."}
            ]
        }

    return output


def get_data():
    cwd = os.getcwd()
    df = pd.read_csv(cwd+"/UI/data/full_set.csv")

    # converts strings to arrays
    df['Classes'] = df['Classes'].apply(ast.literal_eval)
    df['Semester Grades'] = df['Semester Grades'].apply(ast.literal_eval)
    df['Semester Points'] = df['Semester Points'].apply(ast.literal_eval)
    df['CRN'] = df['CRN'].apply(ast.literal_eval)

    return df


def preprocess_and_split_data(df, target_class, min_class_count):

    
    # read the column name 
    columns_file_path = f"./UI/column_names/passfail_for_{target_class}.json"
    with open(columns_file_path, 'r') as f:
        X_train = json.load(f)  # Load column names from JSON

    
    # Return processed data
    return X_train

# Example usage
def process_request(request):
    df = get_data()

    # Call the preprocessing function to get datasets
    X_train = preprocess_and_split_data(df, request['option'], 20)
    # Feature names (from training data)
    feature_names = X_train
    
    # add the current working directory to Python path
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.append(cwd+'/UI/')
    
    # Path to the model, change it to your model
    if request['predictType'] == 'grade':
        model_path = f"./UI/models/multiclass_for_{request['option']}.h5"
    else:
        # I changed the path
        model_path = cwd + f"/UI/models/passfail_for_{request['option']}.h5"
    
    
    result = ""
    if os.path.exists(model_path):
        result = process_advisor_request(request, model_path, feature_names, X_train, request['predictType'])
    else:
        print(model_path)
        print("Error with accesing model")
    
    # Process the request
    print(json.dumps(result, indent=4))
    
    return result

# SHAP output

# import shap
# X_test_1 = X_test.replace(4.0, 8.0)[:50]
# explainer = shap.Explainer(xgb_classifier)

# shap_values = explainer(X_test_1)

# class_index = 3
# shap_values_class = shap_values.values[..., class_index]

# shap.summary_plot(shap_values_class, X_test_1, show=True)

# instance_index = 1

# class_index = 3
# shap_values_class = shap_values.values[..., class_index]

# # Extract the SHAP values and feature values for the specific instance
# shap_values_instance = shap_values_class[instance_index]
# X_test_instance = X_test.iloc[instance_index]

# # Plot the SHAP decision plot for the instance
# shap.decision_plot(
#     base_value=explainer.expected_value[class_index],
#     shap_values=shap_values_instance,
#     features=X_test_instance,
#     ignore_warnings=True
# )


# request = {
#     "data": [
#     {"class": "CHM2046", "grade": 3.67},
#     {"class": "CHM2045", "grade": 3.89},
#     {"class": "LIS4785", "grade": 3.45},
#     {"class": "ANT3610", "grade": 2.67},
#     {"class": "MHS4703", "grade": 2.54},
#     {"class": "FRE1120", "grade": 4.0},
#     {"class": "SPC3710", "grade": 3.12},
#     {"class": "EGN3311", "grade": 2.78},
#     {"class": "LAH2020", "grade": 3.45},
#     {"class": "MUL3011", "grade": 3.67},
#     {"class": "BSC2094C", "grade": 2.34},
#     {"class": "PHI4320", "grade": 3.89},
#     {"class": "AFR1101", "grade": 1.98},
#     {"class": "WST2250", "grade": 4.0},
#     {"class": "ARH2051", "grade": 2.45},
#     {"class": "ISS3420", "grade": 3.56},
#     {"class": "POS2041", "grade": 3.12},
#     {"class": "ENG4674", "grade": 2.23},
#     {"class": "BSC4057", "grade": 3.67},
#     {"class": "LDR4204", "grade": 2.78}
#     ],
#     'option': 'CHM2210',
#     'highschoolGPA': '3.0',
#     'satScore': 1200,
#     'predictType': 'pass_fail',
# }
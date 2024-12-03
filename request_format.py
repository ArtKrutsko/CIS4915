import json
import joblib
import numpy as np
import ast
from sklearn.model_selection import train_test_split
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt
import seaborn as sns


def extract_feature_names(advisor_request, additional_features=['HS GPA', 'Converted_SAT']):
    # Extract class names from the advisor's request
    class_names = [item['class'] for item in advisor_request['data']]
    # Append additional features like HS GPA and SAT
    return class_names + additional_features

def process_advisor_request(request, model_path, feature_names):
    """
    Process the advisor request, create an input vector, and return predictions.
    
    Parameters:
    - request (dict): Input request with class grades and the target class.
    - model_path (str): Path to the saved model for the target class.
    - feature_names (list): List of feature names used in the training data.
    
    Returns:
    - dict: Formatted output with prediction results.
    """
    # Extract data from the request
    class_data = request['data']
    target_class = request['option']
    
    # Initialize the feature vector with default values (-1 for missing classes)
    input_vector = [-1] * len(feature_names)
    
    # Populate the feature vector with grades from the request
    for item in class_data:
        class_name = item['class']
        grade = float(item['grade'])
        if class_name in feature_names:
            index = feature_names.index(class_name)
            input_vector[index] = grade

    # Add SAT/ACT scores if provided
    sat_score = request.get('SAT', -1)  # Default to -1 if not provided
    act_score = request.get('ACT', -1)  # Default to -1 if not provided
    if "SAT" in feature_names:
        input_vector[feature_names.index("SAT")] = sat_score
    if "ACT" in feature_names:
        input_vector[feature_names.index("ACT")] = act_score

    # Load the saved model
    model = joblib.load(model_path)
    
    # Predict using the model
    input_array = np.array(input_vector).reshape(1, -1)  # Reshape for prediction
    prediction = model.predict(input_array)[0]
    
    class_converting = {
        0: 'A+/A/A-/S',
        1: 'B+/B/B-',
        2: 'C+/C/C-',
        3: 'Fail'
    }
    
    # Get Feature Importances
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": X_dev.columns,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    # Plot and save feature importances
    feature_importance_path = f"images/feature_importance_{target_class}.png"
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=importance_df.head(20))
    plt.title(f"Top 20 Feature Importances for {target_class}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(feature_importance_path)
    plt.close()

    # Format the output
    output = {
        "output": [
            {"type": "text", "content": f"Predicted grade for {target_class} is: {class_converting[prediction]}"},
            {"type": "text", "content": f"Top 20 Features by Importance:\n{importance_df.head(20)}"},
            {"type": "image", "content": feature_importance_path}
        ]
    }
    
    return output


def get_data():
    df = pd.read_csv("data/full_set.csv")

    # converts strings to arrays
    df['Classes'] = df['Classes'].apply(ast.literal_eval)
    df['Semester Grades'] = df['Semester Grades'].apply(ast.literal_eval)
    df['Semester Points'] = df['Semester Points'].apply(ast.literal_eval)
    df['CRN'] = df['CRN'].apply(ast.literal_eval)

    return df


def preprocess_and_split_data(df, target_class, min_class_count):
    # Filter for students who took the target class
    pidms_with_target_class = df[df['Classes'].apply(lambda x: target_class in x)]['Pidm'].unique()
    df = df[df['Pidm'].isin(pidms_with_target_class)]
    df = df[['Pidm', 'Semester', 'HS GPA', 'Converted_SAT', 'Semester Points', 'Semester Grades', 'CRN', 'Classes']]

    # Find the first semester when the target class was taken
    def find_first_semester(student_df):
        target_row = student_df[student_df['Classes'].apply(lambda x: target_class in x)]
        if not target_row.empty:
            return target_row['Semester'].min()
        return None

    first_semester = df.groupby('Pidm').apply(find_first_semester).rename('Target_Semester')
    df = df.merge(first_semester, on='Pidm')

    # Filter all semesters before the target class was taken
    filtered_df = df[df['Semester'] <= df['Target_Semester']]

    # Extract grades and points for the target class
    def find_class_grades(student_df):
        for _, row in student_df.iterrows():
            if target_class in row['Classes']:
                index = row['Classes'].index(target_class)
                return row['Semester Points'][index], row['Semester Grades'][index]
        return None, None

    class_grades = filtered_df.groupby('Pidm').apply(find_class_grades).apply(pd.Series)
    class_grades.columns = ['Target_Points', 'Target_Grade']
    filtered_df = filtered_df.merge(class_grades, on='Pidm')

    # Remove rows with invalid grades
    filtered_df = filtered_df[~filtered_df['Target_Grade'].isin(['WE', 'IF', 'W', 'WC'])]
    filtered_df = filtered_df[filtered_df['Semester'] < filtered_df['Target_Semester']]

    # Aggregate data by student
    groupped_df = filtered_df.groupby('Pidm').agg({
        "HS GPA": 'first',
        'Converted_SAT': 'first',
        'Semester Grades': lambda x: sum(x, []),
        'Semester Points': lambda x: sum(x, []),
        'Classes': lambda x: sum(x, []),
        'CRN': lambda x: sum(x, []),
        'Target_Grade': 'first',
        'Target_Points': 'first',
    }).reset_index()

    # Create one-hot encoding for all classes
    all_classes = sorted(set(chain.from_iterable(groupped_df['Classes'])))

    def create_one_hot(classes, points, all_classes):
        one_hot_vector = [-1] * len(all_classes)
        for class_name, point in zip(classes, points):
            if class_name in all_classes:
                one_hot_vector[all_classes.index(class_name)] = point
        return one_hot_vector

    groupped_df['One_Hot_Classes'] = groupped_df.apply(
        lambda row: create_one_hot(row['Classes'], row['Semester Points'], all_classes), axis=1
    )

    one_hot_df = pd.DataFrame(groupped_df['One_Hot_Classes'].tolist(), columns=all_classes, index=groupped_df['Pidm'])

    # Split into train, dev, and test sets
    train, testing_data = train_test_split(one_hot_df, test_size=0.2, random_state=50)
    dev, test = train_test_split(testing_data, test_size=0.5, random_state=50)

    train_set = one_hot_df[one_hot_df.index.isin(train.index)]
    dev_set = one_hot_df[one_hot_df.index.isin(dev.index)]
    test_set = one_hot_df[one_hot_df.index.isin(test.index)]

    # Remove features with fewer than min_class_count observations
    columns_to_remove = []
    for column in train_set.columns:
        value_counts = train_set[column].value_counts()
        max_count = value_counts.max()
        non_max_count = value_counts.sum() - max_count
        if non_max_count <= min_class_count:
            columns_to_remove.append(column)

    train_set = train_set.drop(columns=columns_to_remove)
    dev_set = dev_set.drop(columns=columns_to_remove)
    test_set = test_set.drop(columns=columns_to_remove)

    # Integrate additional features
    train_set = train_set.join(groupped_df.set_index('Pidm')[['HS GPA', 'Converted_SAT', 'Target_Grade']])
    dev_set = dev_set.join(groupped_df.set_index('Pidm')[['HS GPA', 'Converted_SAT', 'Target_Grade']])
    test_set = test_set.join(groupped_df.set_index('Pidm')[['HS GPA', 'Converted_SAT', 'Target_Grade']])

    # Map grades to class labels
    grade_mapping = {
    'A+': 0, 'A': 0, 'A-': 0, 'S': 0,  # Class 0: A
    'B+': 1, 'B': 1, 'B-': 1,  # Class 1: B
    'C+': 2, 'C': 2, 'C-': 2,  # Class 2: C
    'D+': 3, 'D': 3, 'D-': 3, 'F': 3, 'U': 3  # Class 3: Fail
    }

    train_set['Target_Class'] = train_set['Target_Grade'].map(grade_mapping)
    dev_set['Target_Class'] = dev_set['Target_Grade'].map(grade_mapping)
    test_set['Target_Class'] = test_set['Target_Grade'].map(grade_mapping)

    # Drop rows with missing target classes
    train_set.dropna(subset=['Target_Class'], inplace=True)
    dev_set.dropna(subset=['Target_Class'], inplace=True)
    test_set.dropna(subset=['Target_Class'], inplace=True)

    # Separate features and targets
    X_train = train_set.drop(columns=['Target_Grade', 'Target_Class'])
    X_dev = dev_set.drop(columns=['Target_Grade', 'Target_Class'])
    X_test = test_set.drop(columns=['Target_Grade', 'Target_Class'])
    y_train = train_set['Target_Class'].astype(int)
    y_dev = dev_set['Target_Class'].astype(int)
    y_test = test_set['Target_Class'].astype(int)

    # Return processed data
    return X_train, y_train, X_dev, y_dev, X_test, y_test

# Example usage
if __name__ == "__main__":
    request = {
        "data": [
        {"class": "CHM2046", "grade": 3.67},
        {"class": "CHM2045", "grade": 3.89},
        {"class": "LIS4785", "grade": 3.45},
        {"class": "ANT3610", "grade": 2.67},
        {"class": "MHS4703", "grade": 2.54},
        {"class": "FRE1120", "grade": 4.0},
        {"class": "SPC3710", "grade": 3.12},
        {"class": "EGN3311", "grade": 2.78},
        {"class": "LAH2020", "grade": 3.45},
        {"class": "MUL3011", "grade": 3.67},
        {"class": "BSC2094C", "grade": 2.34},
        {"class": "PHI4320", "grade": 3.89},
        {"class": "AFR1101", "grade": 1.98},
        {"class": "WST2250", "grade": 4.0},
        {"class": "ARH2051", "grade": 2.45},
        {"class": "ISS3420", "grade": 3.56},
        {"class": "POS2041", "grade": 3.12},
        {"class": "ENG4674", "grade": 2.23},
        {"class": "BSC4057", "grade": 3.67},
        {"class": "LDR4204", "grade": 2.78}
        ],
        'option': 'CHM2210',
        'SAT': 1200,
        'ACT': 28
    }
    
    df = get_data()

    # Call the preprocessing function to get datasets
    X_train, y_train, X_dev, y_dev, X_test, y_test = preprocess_and_split_data(df, request['option'], 20)
    
    # Feature names (from training data)
    feature_names = X_train.columns.tolist()
    
    # Path to the model
    model_path = f"./models/MLP_model_for_{request['option']}.h5"
    
    # Process the request
    result = process_advisor_request(request, model_path, feature_names)
    print(json.dumps(result, indent=4))

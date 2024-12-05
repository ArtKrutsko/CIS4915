import ast
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

pd.set_option('display.max_columns', None)



# Outputs the sf in a readable format for the prep_data function
def get_data():
    df = pd.read_csv("data/full_set.csv")
    df.shape

    # converts strings to arrays
    df['Classes'] = df['Classes'].apply(ast.literal_eval)
    df['Semester Grades'] = df['Semester Grades'].apply(ast.literal_eval)
    df['Semester Points'] = df['Semester Points'].apply(ast.literal_eval)
    df['CRN'] = df['CRN'].apply(ast.literal_eval)

    return df

# Creates One Hot Encoding, applies undersampling, and outputs split data
def prep_data(df, target_class, min_class_count):
    TARGET_CLASS = target_class
    
    Pidms_with_TARGET_CLASS = df[df['Classes'].apply(lambda x: TARGET_CLASS in x)]['Pidm'].unique()
    df = df[df['Pidm'].isin(Pidms_with_TARGET_CLASS)]
    df = df[['Pidm', 'Semester', 'HS GPA', 'Converted_SAT', 'Semester Points', 'Semester Grades', 'CRN', 'Classes']]
    
    def find_first_semester(student_df):
        chm2210_row = student_df[student_df['Classes'].apply(lambda x: TARGET_CLASS in x)]
        if not chm2210_row.empty:
            return chm2210_row['Semester'].min()
        return None
    
    first_semester = df.groupby('Pidm').apply(lambda x: find_first_semester(x)).rename('Target_Semester')
    df = df.merge(first_semester, on='Pidm')
    
    filtered_df = df[df['Semester'] <= df['Target_Semester']]
    
    def find_class_grades(student_df):
        for _, row in student_df.iterrows():
            if TARGET_CLASS in row['Classes']:
                index = row['Classes'].index(TARGET_CLASS)
                return row['Semester Points'][index], row['Semester Grades'][index]
        return None, None
    
    class_grades = filtered_df.groupby('Pidm').apply(lambda x: find_class_grades(x)).apply(pd.Series)
    class_grades.columns = ['Target_Points', 'Target_Grade']
    
    final_df = filtered_df.merge(class_grades, on='Pidm')
    
    final_df = final_df[~final_df['Target_Grade'].isin(['WE', 'IF', 'W', 'WC'])]
    
    final_df = final_df[final_df['Semester'] < final_df['Target_Semester']]
    groupped_df = final_df.groupby('Pidm').agg({
        "HS GPA": 'first',
        'Converted_SAT': 'first',
        'Semester Grades': lambda x: sum(x, []),
        'Semester Points': lambda x: sum(x, []),
        'Classes': lambda x: sum(x, []),
        'CRN': lambda x: sum(x, []),
        'Target_Grade': 'first',
        'Target_Points': 'first',
    }).reset_index()
    
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
    
    train, testing_data = train_test_split(one_hot_df, test_size=0.2, random_state=50)
    dev, test = train_test_split(testing_data, test_size=0.5, random_state=50)
    
    train_set = one_hot_df[one_hot_df.index.isin(train.index)]
    dev_set = one_hot_df[one_hot_df.index.isin(dev.index)]
    test_set = one_hot_df[one_hot_df.index.isin(test.index)]
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
    
    print(train_set.shape, dev_set.shape, test_set.shape)

    def categorize_grade(grade):
        fail_grades = ['F', 'IF', 'W', 'D-', 'F', 'D+', 'D#', 'D+', 'F#', 'D', 'D', 'D-', 'U', 'W', 
                    'F*', 'D*', 'CF', 'I', 'FF', 'Z', 'W*', 'F+', 'F-', 'F#', 'F*', 'D-*', 'IF', 
                    'IF*', 'D+*', 'CIF', 'Z*', 'IU', 'M', 'CI', 'MU', 'U*', 'ID', 'IB', 'IU*', 
                    'IS', 'CW']
        
        # Categorize the grade into one of the four classes
        if grade in fail_grades:
            return 'Fail'
        elif grade.startswith('A'):
            return 'A class'
        elif grade.startswith('B'):
            return 'B class'
        elif grade.startswith('C'):
            return 'C class'
        else:
            return 'Fail'
    
    groupped_df_filtered = groupped_df[groupped_df['Pidm'].isin(train_set.index)]
    add_columns = ['HS GPA', 'Converted_SAT', 'Target_Grade']
    groupped_df_filtered.set_index('Pidm', inplace=True)
    train_set = train_set.join(groupped_df_filtered[add_columns], )
    
    groupped_df_filtered = groupped_df[groupped_df['Pidm'].isin(dev_set.index)]
    add_columns = ['HS GPA', 'Converted_SAT', 'Target_Grade']
    groupped_df_filtered.set_index('Pidm', inplace=True)
    dev_set = dev_set.join(groupped_df_filtered[add_columns], )

    groupped_df_filtered = groupped_df[groupped_df['Pidm'].isin(test_set.index)]
    add_columns = ['HS GPA', 'Converted_SAT', 'Target_Grade']
    groupped_df_filtered.set_index('Pidm', inplace=True)
    test_set = test_set.join(groupped_df_filtered[add_columns], )
    
    
    train_set['grade_category'] = train_set['Target_Grade'].apply(categorize_grade)
    dev_set['grade_category'] = dev_set['Target_Grade'].apply(categorize_grade)
    test_set['grade_category'] = test_set['Target_Grade'].apply(categorize_grade)
    
    train_set = pd.get_dummies(train_set, columns=['grade_category'])
    dev_set = pd.get_dummies(dev_set, columns=['grade_category'])
    test_set = pd.get_dummies(test_set, columns=['grade_category'])
    
    for col in ['grade_category_A class', 'grade_category_B class', 'grade_category_C class', 'grade_category_Fail']:
        if col not in train_set:
            train_set[col] = 0
        if col not in dev_set:
            dev_set[col] = 0
        if col not in test_set:
            test_set[col] = 0
    
    X_train = train_set.drop(columns=['grade_category_A class', 'grade_category_B class', 'grade_category_C class', 'grade_category_Fail', 'Target_Grade'])
    X_dev = dev_set.drop(columns=['grade_category_A class', 'grade_category_B class', 'grade_category_C class', 'grade_category_Fail', 'Target_Grade'])
    X_test = test_set.drop(columns=['grade_category_A class', 'grade_category_B class', 'grade_category_C class', 'grade_category_Fail', 'Target_Grade'])

    X_train = X_train.dropna()
    X_dev = X_dev.dropna()
    X_test = X_test.dropna()
    
    X_train = X_train.astype('float32')
    X_dev = X_dev.astype('float32')
    X_test = X_test.astype('float32')
    
    y_train = train_set.loc[X_train.index, ['grade_category_A class', 'grade_category_B class', 'grade_category_C class', 'grade_category_Fail']].values.astype('float32')
    y_dev = dev_set.loc[X_dev.index, ['grade_category_A class', 'grade_category_B class', 'grade_category_C class', 'grade_category_Fail']].values.astype('float32')
    y_test = test_set.loc[X_test.index, ['grade_category_A class', 'grade_category_B class', 'grade_category_C class', 'grade_category_Fail']].values.astype('float32')


    return X_train, X_dev, X_test, y_train, y_dev, y_test
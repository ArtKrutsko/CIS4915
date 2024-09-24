import pandas as pd
import numpy as np

student_data = pd.read_csv("ML - Curricular Analytics - PIDM ONLY & Fixed Repeat IND.csv", low_memory=False)
grades = pd.read_csv("parsed_grades.csv")

# Merge the two dataframes to bring in the Quality Points and whether to count in GPA
student_data = pd.merge(student_data, grades[['Code', 'Quality Points', 'Count in GPA?']], 
              left_on='FINAL_GRADE', right_on='Code', how='left', suffixes=('', '_grades'))

# Fill missing 'Quality Points' with 0.0 for unrecognized grades
student_data['Quality Points'] = student_data['Quality Points'].fillna(0.0)
student_data['Count_in_GPA'] = student_data['Count in GPA?'] == 'Y'

#Final and Semester GPAs (Assuming all classes are equal credits)
student_data['Valid_Grades'] = np.where(student_data['Count in GPA?'] == 'Y' , student_data['Quality Points'], np.nan)
student_final_gpa = student_data.groupby('Pidm')['Valid_Grades'].mean().reset_index()
student_data = student_data.merge(student_final_gpa, on='Pidm', how='left', suffixes=('', '_mean'))
student_data.rename(columns={'Valid_Grades_mean':'Final GPA'}, inplace=True)

student_semester_gpa = student_data.groupby(['Pidm', 'Term'])['Valid_Grades'].mean().reset_index()
student_data = student_data.merge(student_semester_gpa, on=['Pidm', 'Term'], how='left', suffixes=('', '_mean'))
student_data.rename(columns={'Valid_Grades_mean':'Semester GPA'}, inplace=True)

#Student Classes & Points per Semester (As an array of strings)
student_data['class'] = (student_data['SUBJ'] + student_data['CRSE_NUMB']).astype(str)
semester_classes = student_data.groupby(['Pidm', 'Term']).agg({
    'FINAL_GRADE': list,
    'Quality Points': list, 
    'class': list
}).reset_index()

#Drop unecessary columns
student_data.drop(['CRN', 'SUBJ', 'CRSE_NUMB', 'REPEAT_IND', 'FINAL_GRADE', 'class', 'Code', 'Count in GPA?', 'Count_in_GPA', 'Valid_Grades'], axis=1, inplace=True)

#Remove repeated rows in the demographic columns
student_data = student_data.groupby(['Pidm', 'Term']).agg({ 
    'Admit_Code': 'first', 
    'Admit_Level': 'first', 
    'Admit_College': 'first', 
    'Admit_Major_Code': 'first', 
    'Major_Desc': 'first', 
    'Trump_Race': 'first', 
    'Trump_Race_Desc': 'first', 
    'MULTI': 'first', 
    'Race': 'first', 
    'NEW_ETHNICITY': 'first', 
    'GENDER_Code': 'first', 
    'GENDER': 'first', 
    'CITZ_IND': 'first', 
    'CITZ_CODE': 'first', 
    'CITZ_DESC': 'first', 
    'Final_GPA': 'first', 
    'ACTE': 'first', 
    'ACTM': 'first', 
    'ACTR': 'first', 
    'ACTS': 'first', 
    'EACT': 'first', 
    'SAT-ERW': 'first', 
    'SATM': 'first', 
    'SAT_TOTAL': 'first', 
    'Final GPA': 'first',
    'Semester GPA': 'first'
}).reset_index()

#Add Semester grades and gpa points to df
student_data = student_data.merge(semester_classes[['Pidm', 'Term', 'FINAL_GRADE', 'Quality Points', 'class']], on=['Pidm', 'Term'], how='left')
student_data.rename(columns={'Final_GPA':'HS GPA', 'Term':'Semester','FINAL_GRADE':'Semester Grades', 'Quality Points':'Semester Points', 'class':'Classes'}, inplace=True)

#Correct datatypes and output to csv
student_data = student_data.astype(str)
student_data['Pidm'] = student_data['Pidm'].astype(int)
student_data['Final GPA'] = student_data['Final GPA'].astype(float).round(2)
student_data['Semester GPA'] = student_data['Semester GPA'].astype(float).round(2)
student_data['Semester'] = student_data['Semester'].astype(float).astype(int).astype(str)
student_data.to_csv("formatted_data.csv", index=False)
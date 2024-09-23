import pandas as pd

student_data = pd.read_csv("ML - Curricular Analytics - PIDM ONLY & Fixed Repeat IND.csv")
grades = pd.read_csv("parsed_grades.csv")
student_data.shape

#Add Numeric GPA Column
grade_mapping = dict(grades[['Code', 'Quality Points']].values)
student_data = student_data.drop(student_data[(student_data.FINAL_GRADE == 'W') | 
                                              (student_data.FINAL_GRADE == 'S') | 
                                              (student_data.FINAL_GRADE == 'U') | 
                                              (student_data.FINAL_GRADE.isna()) |
                                              (student_data.FINAL_GRADE == '')].index)
student_data['numeric_gpa'] = student_data.FINAL_GRADE.map(grade_mapping)
student_data['numeric_gpa'] = student_data['numeric_gpa'].astype(float)

#Final and Semester GPAs (Assuming all classes are equal credits)
student_final_gpa = student_data.groupby('Pidm')['numeric_gpa'].mean().reset_index()
student_semester_gpa = student_data.groupby(['Pidm', 'Term'])['numeric_gpa'].mean().reset_index()

#Student Classes per Semester (As an array of strings)
student_data['class'] = (student_data['SUBJ'] + student_data['CRSE_NUMB']).astype(str)
semester_classes = student_data.groupby(['Pidm', 'Term']).agg({
    'FINAL_GRADE': list,
    'class': list
}).reset_index()

#Map Final GPA onto main df and drop unecessary columns
student_gpa_map = dict(student_final_gpa[['Pidm', 'numeric_gpa']].values)
student_data['Final GPA'] = student_data.Pidm.map(student_gpa_map)
student_data.drop(['CRN', 'SUBJ', 'CRSE_NUMB', 'REPEAT_IND', 'FINAL_GRADE', 'numeric_gpa', 'class'], axis=1, inplace=True)

#Eliminated repeated demographic data & Add semester gpa & classes
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
    'Final GPA': 'first'
}).reset_index()

student_data['Semester GPA'] = student_data.merge(student_semester_gpa[['Pidm', 'Term', 'numeric_gpa']], on=['Pidm', 'Term'], how='left')['numeric_gpa']
student_data = student_data.merge(semester_classes[['Pidm', 'Term', 'FINAL_GRADE', 'class']], on=['Pidm', 'Term'], how='left')
student_data.rename(columns={'Final_GPA':'HS GPA', 'Term':'Semester','FINAL_GRADE':'Semester Grades', 'class':'Classes'}, inplace=True)

#Correct Datatypes and output to .csv
student_data = student_data.astype(str)
student_data['Pidm'] = student_data['Pidm'].astype(int)
student_data['Final GPA'] = student_data['Final GPA'].astype(float).round(2)
student_data['Semester GPA'] = student_data['Semester GPA'].astype(float).round(2)
student_data['Semester'] = student_data['Semester'].astype(float).astype(int).astype(str)
student_data.to_csv("formatted_data.csv", index=False)
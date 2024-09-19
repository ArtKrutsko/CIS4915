import pandas as pd

student_data = pd.read_csv("studentData.csv")
grades = pd.read_csv("parsed_grades.csv")

#Add Numeric GPA Column

grade_mapping = dict(grades[['Code', 'Quality Points']].values)

student_data = student_data.drop(student_data[student_data.FINAL_GRADE == 'W'].index)
student_data['numeric_gpa'] = student_data.FINAL_GRADE.map(grade_mapping)
student_data['numeric_gpa'] = student_data['numeric_gpa'].astype(float)

#Final and Semester GPAs (Assuming all classes are equal credits)

student_final_gpa = student_data.groupby('Pidm')['numeric_gpa'].mean().reset_index()
student_semester_gpa = student_data.groupby(['Pidm', 'Term'])['numeric_gpa'].mean().reset_index()

#Student Classes per Semester (As an array of strings)

student_data['class'] = student_data['SUBJ'] + student_data['CRSE_NUMB']
student_data['class'] = student_data['class'].astype(str)
semester_classes = student_data.groupby(['Pidm', 'Term'])['class'].apply(list).reset_index()

student_gpa_map = dict(student_final_gpa[['Pidm', 'numeric_gpa']].values)
student_data['Final GPA'] = student_data.Pidm.map(student_gpa_map)

student_data.drop(['CRN', 'SUBJ', 'CRSE_NUMB', 'REPEAT_IND', 'FINAL_GRADE', 'numeric_gpa', 'class'], axis=1, inplace=True)
student_data.rename(columns={'Final_GPA':'HS GPA', 'Term':'Semester'}, inplace=True)

student_data = student_data.groupby(['Pidm', 'Admit_Code', 'Admit_Desc', 'Admit_Term', 'Admit_College', 'Admit_Major_Code', 'Major_Desc', 'MULTI', 'Race', 'NEW_ETHNICITY', 'GENDER_Code', 'GENDER', 'CITZ_IND', 'CITZ_CODE', 'CITZ_DESC', 'HS GPA', 'ACTE', 'ACTM', 'ACTR', 'ACTS', 'EACT', 'SAT-ERW', 'SATM', 'SAT_TOTAL', 'Semester', 'Final GPA']).size().reset_index(name='count')
student_data.drop(['count'], axis=1, inplace=True)
student_data['Semester GPA'] = student_semester_gpa['numeric_gpa']
student_data['Classes'] = semester_classes['class']

student_data = student_data.astype(str)
student_data['Pidm'] = student_data['Pidm'].astype(int)
student_data['Final GPA'] = student_data['Final GPA'].astype(float)
student_data['Semester'] = student_data['Semester'].astype(float)
student_data['Semester'] = student_data['Semester'].round(0)
student_data['Semester'] = student_data['Semester'].astype(str)
student_data.to_csv('new_table.csv')






print(student_data)
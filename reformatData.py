from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

student_data = pd.read_csv("ML - Curricular Analytics - PIDM ONLY & Fixed Repeat IND.csv", low_memory=False)
grades = pd.read_csv("parsed_grades.csv")

student_data = student_data.drop(columns=['Admit_Term', 'Admit_Major_Code'])
student_data = student_data.dropna(subset=['Term', 'CRN', 'SUBJ', 'CRSE_NUMB', 'FINAL_GRADE'])

#Save latest student major
student_data['Term'] = student_data['Term'].astype(int)

last_sem_idx = student_data.groupby('Pidm')['Term'].idxmax()
latest_majors = student_data.loc[last_sem_idx, ['Pidm', 'Major_Desc']]
majors_dict = latest_majors.set_index('Pidm')['Major_Desc'].to_dict()
student_data['Lastest_Major'] = student_data['Pidm'].map(majors_dict)
student_data = student_data.drop(columns=['Major_Desc'])
student_data.drop_duplicates(subset=['Pidm', 'Term', 'CRN'], inplace=True)


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
    'class': list,
    'CRN': list,
}).reset_index()

#Drop unecessary columns
student_data.drop(['SUBJ', 'REPEAT_IND', 'FINAL_GRADE', 'class', 'Code', 'Count in GPA?', 'Count_in_GPA', 'Valid_Grades'], axis=1, inplace=True)

#Remove repeated rows in the demographic columns
student_data = student_data.groupby(['Pidm', 'Term']).agg({ 
    'Admit_Level': 'first', 
    'Admit_College': 'first',
    'Lastest_Major': 'first',
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
student_data = student_data.merge(semester_classes[['Pidm', 'Term', 'FINAL_GRADE', 'Quality Points', 'class', 'CRN']], on=['Pidm', 'Term'], how='left')
student_data.rename(columns={'Final_GPA':'HS GPA', 'Term':'Semester','FINAL_GRADE':'Semester Grades', 'Quality Points':'Semester Points', 'class':'Classes'}, inplace=True)

###Code from Varma to convert SATs
# List of score columns
score_columns = ['ACTE', 'ACTM', 'ACTR', 'ACTS', 'EACT', 'SAT-ERW', 'SATM', 'SAT_TOTAL', 'HS GPA']

# Convert score columns to numeric
for col in score_columns:
    student_data[col] = pd.to_numeric(student_data[col], errors='coerce')

# ACT to SAT conversion table
act_to_sat_conversion = {
    36: 1590, 35: 1540, 34: 1500, 33: 1460, 32: 1430, 31: 1400,
    30: 1370, 29: 1340, 28: 1310, 27: 1280, 26: 1240, 25: 1210,
    24: 1180, 23: 1140, 22: 1110, 21: 1080, 20: 1040, 19: 1010,
    18: 970, 17: 930, 16: 890, 15: 850, 14: 800, 13: 760,
    12: 710, 11: 670, 10: 630, 9: 590
}

# Convert EACT to SAT
def convert_act_to_sat(eact_score):
    if pd.isna(eact_score):
        return np.nan
    return act_to_sat_conversion.get(int(eact_score), np.nan)

# Apply the conversion to EACT scores
student_data['Converted_SAT'] = student_data['SAT_TOTAL']

# Identify where SAT_TOTAL is missing but EACT is available
mask = student_data['SAT_TOTAL'].isna() & student_data['EACT'].notna()

# Apply conversion
student_data.loc[mask, 'Converted_SAT'] = student_data.loc[mask, 'EACT'].apply(convert_act_to_sat)

# Step 4: Handle Remaining Missing Values

# Drop rows where Converted_SAT or Final_GPA is still NaN
student_data = student_data.dropna(subset=['Converted_SAT', 'HS GPA'])

#Correct datatypes and output to csv
student_data = student_data.astype(str)
student_data['Pidm'] = student_data['Pidm'].astype(int)
student_data['Final GPA'] = student_data['Final GPA'].astype(float).round(2)
student_data['Semester GPA'] = student_data['Semester GPA'].astype(float).round(2)
student_data['Semester'] = student_data['Semester'].astype(float).astype(int).astype(str)

#Split into train-dev-test sets by Pidm
student_ids = student_data['Pidm'].unique()
train, testing_data = train_test_split(student_ids, test_size=0.2, random_state=50)
dev, test = train_test_split(testing_data, test_size=0.5, random_state=50)
train_set = student_data[student_data['Pidm'].isin(train)]
dev_set = student_data[student_data['Pidm'].isin(dev)]
test_set = student_data[student_data['Pidm'].isin(test)]

#Output to .csv
student_data.to_csv("full_set.csv", index=False)
train_set.to_csv("train_set.csv", index=False)
dev_set.to_csv("dev_set.csv", index=False)
test_set.to_csv("test_set.csv", index=False)
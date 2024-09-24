import pandas as pd

grades = pd.read_csv("data/SHAGRDE.csv")

# Leave only grades from UG level
grades_UG = grades[grades["Level"] == "UG"]

# Leave the grade GPA from the latest update(column "Term")
grades_UG_Latest = grades_UG.loc[grades_UG.groupby('Code')['Term'].idxmax()]

# Leave where GPA is counted
grades_final_1 = grades_UG_Latest[(grades_UG_Latest["Traditional Ind"] == "Y")]

grades_final_1.to_csv('data/parsed_grades.csv', index=False)
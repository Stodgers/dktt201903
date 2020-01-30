#%%

import pandas as pd
import numpy as np

#%%

with open("./data/KnowledgeTracing.txt", "r") as f:
    subjectList = [line.strip().split("\t") for line in f.readlines()]

subjectList[0][0] = "student_id"
subjectList[0][1] = "question_id"

#%%

newSubjectList = [item for item in subjectList if len(item) == 6]

#%%

df = pd.DataFrame(newSubjectList[1:], columns=newSubjectList[0])

#%%

df.head(5)

#%%

newDf = df.sort_values(['student_id', 'end_time'])

#%%

newDf.head(5)

#%%

newDf.is_correct.value_counts()

#%%

newDf1 = newDf[newDf["is_correct"].isin(["1", "2", "3"])]

#%%

newDf1["is_correct"] = newDf1["is_correct"].replace(["2", "3"], 0)
newDf1["is_correct"] = newDf1["is_correct"].replace("1", 1)

#%%

newDf1.is_correct.value_counts()

#%%

students = newDf1.student_id.tolist()
len(set(students))

#%%

skills = newDf1.skill_id.tolist()
len(set(skills))

#%%

newDf1.head(5)

#%%

ktDf = newDf1[["student_id", "skill_id", "is_correct"]]

#%%

ktDf.head(5)

#%%

ktDf.to_csv("./data/knowledgeTracing.csv", index=False)

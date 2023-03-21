import pandas as pd

keys = ['MRN','PatientID','DateOfVisit']

df1 = pd.read_excel('study_criteria_table_label_V6.xlsx')
df1['DateOfVisit'] = pd.to_datetime(df1.DateOfVisit)
df1 = df1.drop(columns='MRN')
df1 = df1.rename(columns={'MRN_key':'MRN'})
df1 = df1.dropna(subset=keys).reset_index(drop=True)
df1 = df1.drop_duplicates(subset=keys).reset_index(drop=True)

df2 = pd.read_csv('haoqi_sleep_master.csv')
df2 = df2.rename(columns={'VisitDate':'DateOfVisit'})
df2['DateOfVisit'] = pd.to_datetime(df2.DateOfVisit)
df2 = df2.dropna(subset=keys).reset_index(drop=True)
df2 = df2.drop_duplicates(subset=keys).reset_index(drop=True)

df = df1.merge(df2,on=keys,how='inner',validate='1:1')
cols = ['HashFileName', 'ShiftedVisitDate',
        'Age', 'Sex', 'TypeOfTest',
        'True_Stage', 'True_Certainty', 'True_Diease',
        'Dementia_dT', 'MCI_dT', 'Symptomatic_dT', 'Note',
        'Predicted_Stage', 'Predicted_Certainty', 'Predicted_dT', 'Predicted_Disease',
       ]
df[cols].to_excel('dementia_mastersheet_Elissa_deid.xlsx', index=False)

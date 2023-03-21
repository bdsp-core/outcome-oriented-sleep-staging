import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


df = pd.read_excel('dementia_mastersheet_Elissa_deid.xlsx')
print('original', df.shape)

df.loc[df.Sex=='F', 'Sex'] = 'Female'
df.loc[df.Sex=='Unknown', 'Sex'] = np.nan
df = df.dropna(subset=['HashFileName', 'ShiftedVisitDate', 'Age', 'Sex']).reset_index(drop=True)
print('after drop na', df.shape)
df = df[df.TypeOfTest.str.contains('diag', case=False)].reset_index(drop=True)
print('after diagnositic only', df.shape)
df = df[pd.isna(df.True_Certainty)].reset_index(drop=True)
print('after drop low True_Certainty', df.shape)
#df = df[(pd.isna(df.True_Certainty)|(df.True_Certainty!='Low'))&(pd.isna(df.Predicted_Certainty)|(df.Predicted_Certainty!='Low'))].reset_index(drop=True)
#print('after drop low certainty', df.shape)
df['HashID'] = df.HashFileName.astype(str).str.split('_', expand=True)[0]
df = df.drop_duplicates(subset=['HashID', 'ShiftedVisitDate']).reset_index(drop=True)
print('after drop duplicates', df.shape)

# negate Predicted_dT since it is PSGtime - eventtime, we want eventtime-PSGtime
df.loc[pd.isna(df.Predicted_dT), 'Predicted_dT'] = 0
df.loc[:, 'Predicted_dT'] = -df.loc[:, 'Predicted_dT']
# get True_dT
df['True_dT'] = 0
for i in range(len(df)):
    if df.True_Stage.iloc[i] in ['Dementia', 'MCI', 'Symptomatic']:
        col = df.True_Stage.iloc[i]+'_dT'
        df.loc[i, 'True_dT'] = -df[col].iloc[i]
df.loc[pd.isna(df.True_Stage), 'dT'] = df.loc[pd.isna(df.True_Stage), 'Predicted_dT']
df.loc[pd.notna(df.True_Stage), 'dT'] = df.loc[pd.notna(df.True_Stage), 'True_dT']
df.loc[:,'Age_event'] = df.Age+df.dT/365

# for sure dementia
ids1 = (df.True_Stage=='Dementia')&(df.Predicted_Stage=='Dementia')#&(df.Age_event>50)
# predicted dementia, high certainty
ids2 = pd.isna(df.True_Stage)&(df.Predicted_Stage=='Dementia')&(df.Predicted_Certainty=='High')#&(df.Age_event>50)
ids_dementia = np.where((ids1|ids2)&pd.notna(df.dT)&(df.dT!=0))[0]
df.loc[ids_dementia, 'time_Dementia'] = df.iloc[ids_dementia].dT/365
df.loc[ids_dementia, 'cens_Dementia'] = 0
print(f'len(ids_dementia)={len(ids_dementia)}')

# for sure no dementia, age>50
ids3 = (df.True_Stage=='No Dementia')&np.in1d(df.Predicted_Stage, ['No Dementia', 'Symptomatic'])#&(df.Age_event>50)
# predicted no dementia, high certainty, age>50
ids4 = pd.isna(df.True_Stage)&(df.Predicted_Stage=='No Dementia')&(df.Age_event>50)#&(df.Predicted_Certainty=='High')
# for sure no dementia, 18<age<50
#ids5 = (df.Age_event<50)&(df.Age_event>18)
ids_nodementia = np.where(ids3|ids4)[0]#|ids5

# matching
available = np.ones(len(df), dtype=bool)
np.random.seed(2023)
for i in ids_dementia:
    age = df.Age.iloc[i]
    sex = df.Sex.iloc[i]
    matched_ids = ids_nodementia[(np.abs(df.Age.iloc[ids_nodementia]-age)<=1)&(df.Sex.iloc[ids_nodementia]==sex)&available[ids_nodementia]]
    matched_id  = np.random.choice(matched_ids, 1)
    available[matched_id] = False
ids_nodementia = np.where(~available)[0]
print(ttest_ind(df.Age[ids_dementia],df.Age[ids_nodementia]))
assert len(set(ids_dementia)&set(ids_nodementia))==0

df.loc[ids_nodementia, 'time_Dementia'] = np.nan #TODO time to death
df.loc[ids_nodementia, 'cens_Dementia'] = 1
print(f'len(ids_nodementia)={len(ids_nodementia)}')

df = df.iloc[np.r_[ids_dementia, ids_nodementia]].reset_index(drop=True)    
print('after matching', df.shape)

df = df[['HashFileName', 'ShiftedVisitDate', 'Age', 'Sex', 'cens_Dementia', 'time_Dementia']]
print(df.shape)
df.to_csv('dementia_mastersheet_small.csv', index=False)
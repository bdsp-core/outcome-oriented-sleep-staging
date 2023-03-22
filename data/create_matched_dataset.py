import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from tqdm import tqdm


outcome = 'Dementia'
cens_col = 'cens_'+outcome
time_col = 'time_'+outcome
y_col    = 'Y_'+outcome

df = pd.read_excel('mastersheet_outcome_deid.xlsx')
print('original', df.shape)

df.loc[pd.isna(df.Race), 'Race'] = 'Other'
df = df[np.in1d(df.Race, ['White', 'Black', 'Other', 'Hispanic', 'Asian'])].reset_index(drop=True)
df = df.dropna(subset=['HashID', 'BDSPPatientID', 'DOVshifted', 'Age', 'Sex', 'AHI', 'SignalPath', 'AnnotPath']).reset_index(drop=True)

annot_folder = r'D:\tmp_new_redacted_annotations_to_be_moved-03062023'
good_ids = []
for i in tqdm(range(len(df))):
    annot_path = os.path.join(annot_folder, *df.AnnotPath.iloc[i].split('/')[-2:])
    if os.path.exists(annot_path):
        good_ids.append(i)
df = df.iloc[good_ids].reset_index(drop=True)

ids1 = np.where(pd.isna(df[cens_col])|((df[time_col]>0)&(df[time_col]<=1)&(df[cens_col]==0)))[0]
ids0 = np.where(df[cens_col]==1)[0]
np.random.seed(24)
"""
ids0 = np.random.choice(ids0, len(ids1)*2)
"""
available = np.zeros(len(df), dtype=bool)
available[ids0] = True
matched_ids0 = []
for i in tqdm(ids1):
    age = df.Age.iloc[i]
    sex = df.Sex.iloc[i]
    race = df.Race.iloc[i]
    ahi = df.AHI.iloc[i]
    bmi = df.BMI.iloc[i]
    ids0_ = ids0[(np.abs(df.Age[ids0]-age)<=5) &\
        (df.Sex[ids0]==sex) &\
        (np.abs(df.BMI[ids0]-bmi)<=2) &\
        available[ids0]]
    #    (df.Race[ids0]==race) &\
    #    (np.abs(df.AHI[ids0]-ahi)<=5) &\
    ids0_ = np.random.choice(ids0_)
    matched_ids0.append(ids0_)
    available[ids0_] = False
ids0 = matched_ids0
print(ttest_ind(df.Age[ids0].values, df.Age[ids1].values))
print(ttest_ind(df.BMI[ids0].values, df.BMI[ids1].values))

df.loc[ids0, y_col] = 0
df.loc[ids1, y_col] = 1
df = df.dropna(subset=y_col).reset_index(drop=True)

df = df[[
    'HashID', 'BDSPPatientID', 'DOVshifted', 'Age', 'Sex', 'ESS', 'BMI',
    'Race', 'Ethnicity', 'AHI', 'TypeOfTest', 'MedBenzo', 'MedAntiDep_SSRI',
    'MedAntiDep_SNRI', 'MedAntiDep_TCA', 'MedAntiDep', 'MedSedative',
    'MedAntiEplipetic', 'MedStimulant', y_col, 'SignalPath', 'AnnotPath']]
df['DOVshifted'] = df.DOVshifted.dt.strftime('%Y-%m-%d')
print(df)
df.to_csv(f'mastersheet_matched_{outcome}.csv', index=False)

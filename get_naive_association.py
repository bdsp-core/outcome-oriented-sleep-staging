from itertools import product
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


conditions = ['existing', 'future5y']
outcomes = ['IntracranialHemorrhage', 'IschemicStroke', 'Dementia', 'MCI+Dementia', 'Atrial_Fibrillation', 'Myocardial_Infarction', 'DiabetesII', 'Hypertension', 'Bipolar_Disorder', 'Depression', 'Death']
features = ['NREMPercTST','REMPercTST','SleepEfficiency']

df1 = pd.read_excel('../sleep-outcome-prediction/mastersheet_outcome_deid.xlsx')
df2 = pd.read_csv('../sleep-outcome-prediction/to_be_used_features_NREM_deid.csv')
df  = df1.merge(df2[['HashID']+features],on='HashID',how='left',validate='1:1')

for cond in conditions:
    res = np.zeros((len(outcomes), len(features)))
    for outcome, feature in product(outcomes, features):
        if cond=='existing' and outcome=='Death':
            auc = np.nan
        else:
            x = df[feature].values
            if cond=='existing':
                y = (df['time_'+outcome].values<0).astype(int)
            elif cond=='future5y':
                ids = df['time_'+outcome].values>=0
                y = ((df['cens_'+outcome][ids].values==0)&(df['time_'+outcome][ids].values<=5)).astype(int)
                x = x[ids]
            ids = (~np.isnan(x))&(~np.isnan(y))
            x = x[ids]
            y = y[ids]
            if len(set(y))==1:
                auc = np.nan
            else:
                auc = roc_auc_score(y, x)
            if not np.isnan(auc) and auc<0.5:
                auc = 1-auc
        res[outcomes.index(outcome), features.index(feature)] = auc
    df_res = pd.DataFrame(data=res, columns=features,index=outcomes)
    df_res.to_csv(f'macrostructure_outcome_auc_{cond}.csv')

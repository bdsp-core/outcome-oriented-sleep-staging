import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score

outcome = 'Dementia'
#covs = ['Age', 'Sex', 'Race', 'BMI', 'MedBenzo', 'MedAntiDep', 'MedSedative', 'MedAntiEplipetic', 'MedStimulant']

df = pd.read_csv('../data/mastersheet_small_Dementia.csv')

# get features
stage2num = {'w':5, 'wake':5, 'rem':4, 'r':4, 'n1':3, 'n2':2, 'n3':1}
annot_folder = r'D:\tmp_new_redacted_annotations_to_be_moved-03062023'
for i in tqdm(range(len(df))):
    annot_path = os.path.join(annot_folder, *df.AnnotPath.iloc[i].split('/')[-2:])
    annot = pd.read_csv(annot_path)
    stage_ids = annot.event.str.contains('sleep_stage_', case=False)
    stages = annot.event[stage_ids].str.split('_', expand=True)[2]
    stages = stages.apply(lambda x:stage2num.get(str(x).lower(),np.nan)).values
    df.loc[i, 'SleepEfficiency'] = np.sum(stages<=4)/np.sum(stages<=5)
    df.loc[i, 'NREMPercTST'] = np.sum(stages<=3)/np.sum(stages<=4)
    df.loc[i, 'N3PercTST'] = np.sum(stages==1)/np.sum(stages<=4)
df[['HashID', 'BDSPPatientID', 'DOVshifted', 'Age', 'Sex', 'ESS', 'BMI', 'AHI',
    'Y_Dementia', 'SleepEfficiency', 'NREMPercTST', 'N3PercTST']].to_csv('AASM_macrostructure_features.csv', index=False)

#df = pd.get_dummies(df, columns=['Sex', 'Race'])
#df = df.drop(columns=['Sex_Male', 'Race_White'])
#covs = [x for x in covs if x not in ['Sex', 'Race']]+[x for x in df.columns if x.startswith('Race')]+[x for x in df.columns if x.startswith('Sex')]
yname = 'Y_'+outcomes[0]##

metrics = ['auroc', 'auprc']
random_state = 2023
Nbt = 1000
Ncv = 5
perfs_bt = {}
for feat in features:
    print(feat)
    #if feat=='none':
    Xnames = [feat]#covs
    #else:
    #    Xnames = covs+[feat]

    np.random.seed(random_state)
    perfs_bt[feat] = {x:[] for x in metrics}
    for bti in tqdm(range(Nbt+1)):
        Xbt = df[Xnames].values
        ybt = df[yname].values
        if bti>0:
            btids = np.random.choice(len(df), len(df), replace=True)
            Xbt = Xbt[btids]
            ybt = ybt[btids]

        cvf = KFold(n_splits=Ncv, random_state=random_state+bti, shuffle=True)
        aurocs = []; auprcs = []
        for trids, teids in cvf.split(Xbt, ybt):
            model = LogisticRegression(penalty=None, class_weight='balanced', max_iter=1000)
            model.fit(Xbt[trids], ybt[trids])
            ypte = model.predict_proba(Xbt[teids])[:,1]
            aurocs.append(roc_auc_score(ybt[teids], ypte))
            auprcs.append(average_precision_score(ybt[teids], ypte))
        perfs_bt[feat]['auroc'].append(np.mean(aurocs))
        perfs_bt[feat]['auprc'].append(np.mean(auprcs))

    for x in metrics:
        lb, ub = np.nanpercentile(perfs_bt[feat][x][1:], (2.5, 97.5))
        perfs_bt[feat][x] = (perfs_bt[feat][x][0], lb, ub)

val = np.zeros((len(features), len(metrics))).astype(object)
for fi, feat in enumerate(features):
    for mi, metric in enumerate(metrics):
        v = perfs_bt[feat][metric]
        val[fi,mi] = f'{v[0]:.2f} ({v[1]:.2f}--{v[2]:.2f})'
df_res = pd.DataFrame(data=val, columns=metrics, index=features)
print(df_res)
df_res.to_csv(f'AASM_associations_{outcomes[0]}_Nbt{Nbt}.csv')


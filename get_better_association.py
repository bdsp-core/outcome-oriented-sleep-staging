from itertools import product
import os
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold


outcomes = ['IntracranialHemorrhage', 'IschemicStroke', 'Dementia', 'MCI+Dementia', 'Atrial_Fibrillation', 'Myocardial_Infarction', 'DiabetesII', 'Hypertension', 'Bipolar_Disorder', 'Depression']#TODO, 'Death']
features = ['SleepEfficiency', 'NREMPercTST']#REMPercTST
covs = ['Age', 'Sex', 'Race', 'BMI', 'MedBenzo', 'MedAntiDep', 'MedSedative', 'MedAntiEplipetic', 'MedStimulant']

df1 = pd.read_excel('../sleep-outcome-prediction/mastersheet_outcome_deid.xlsx')
df2 = pd.read_csv('../sleep-outcome-prediction/to_be_used_features_NREM_deid.csv')
df  = df1.merge(df2[['HashID']+features],on='HashID',how='left',validate='1:1')

df.loc[pd.isna(df.Race), 'Race'] = 'Other'
df = df[np.in1d(df.Race, ['White', 'Black', 'Other', 'Hispanic', 'Asian'])].reset_index(drop=True)
df = pd.get_dummies(df, columns=['Sex', 'Race'])
df = df.drop(columns=['Sex_Male', 'Race_White'])
covs = [x for x in covs if x not in ['Sex', 'Race']]+[x for x in df.columns if x.startswith('Race')]+[x for x in df.columns if x.startswith('Sex')]

# CV
cv_folds = np.zeros(len(df))+np.nan
Ncv = 5
cv = StratifiedKFold(n_splits=Ncv, shuffle=True, random_state=2023)
for cvi, (_, teids) in enumerate(cv.split(df[['Age']].values, df.cens_Death)):
    cv_folds[teids] = cvi

res = np.zeros((len(outcomes), len(features)))
for outcome, feature in product(outcomes, features):
    print(outcome, feature)
    if outcome=='Death':
        cols = [feature]+covs+[f'time_{outcome}', f'cens_{outcome}']
        df2 = df[cols]
    else:
        cols = [feature]+covs+[f'time_{outcome}', f'cens_{outcome}','time_Death', 'cens_Death']
        df2 = df[cols]
        df2 = df2.rename(columns={f'time_{outcome}':'time_outcome', f'cens_{outcome}':'cens_outcome'})
    ids = df2.time_outcome>0
    df2 = df2[ids].reset_index(drop=True)
    cv_folds_ = cv_folds[ids]

    # CV
    perfs_cv = []
    for cvi in tqdm(range(Ncv)):
        dftr = df2[cv_folds_!=cvi].reset_index(drop=True)
        dfte = df2[cv_folds_==cvi].reset_index(drop=True)

        dftr_r_path = os.path.join(os.getcwd(), 'dftr.csv')
        dfte_r_path = os.path.join(os.getcwd(), 'dfte.csv')
        dftr.to_csv(dftr_r_path, index=False)
        dfte.to_csv(dfte_r_path, index=False)
        output_r_path = os.path.join(os.getcwd(), 'output.csv')
        code_r_path = os.path.join(os.getcwd(), 'code.R')

        rcode = f"""library(survival)

# read file
df <- read.csv('{dftr_r_path}')
dfte <- read.csv('{dfte_r_path}')

# preprocessing
df$time <- ifelse(df$cens_outcome==1, df$time_Death, df$time_outcome)
df$event <- factor(ifelse(df$cens_outcome==1, 2*(1-df$cens_Death), 1), 0:2, labels=c("censor", "event1", "event2"))
xnames <- names(df)[!names(df) %in% c('time', 'event', 'cens_Death', 'cens_outcome', 'time_Death','time_outcome')]
df$id <- 1:nrow(df)

# fit model
formula_ <- as.formula(paste('Surv(time, event)~', paste(xnames, collapse = "+")))
model <- coxph(formula_, data=df, id=id, ties='breslow')
model_summary <- summary(model)
coef <- cbind(model_summary$coefficients, model_summary$conf.int)
coef <- coef[, c('coef', 'Pr(>|z|)', 'lower .95', 'upper .95')]

# test
dfte$time <- ifelse(dfte$cens_outcome==1, dfte$time_Death, dfte$time_outcome)
dfte$event <- factor(ifelse(dfte$cens_outcome==1, 2*(1-dfte$cens_Death), 1), 0:2, labels=c("censor", "event1", "event2"))
# reproduce coxph.fit$concordance
zp1 <- drop(as.matrix(dfte[,xnames])%*%coef[1:(nrow(coef)/2),'coef'])
zp2 <- drop(as.matrix(dfte[,xnames])%*%coef[(nrow(coef)/2+1):nrow(coef),'coef'])
names(zp1) <- NULL
names(zp2) <- NULL
y2 <- Surv(dfte$time, dfte$event)
y2 <- aeqSurv(y2)
y2 <- Surv(c(y2[,1], y2[,1]),c(as.integer(y2[,2]==1), as.integer(y2[,2]==2)))
zp <- c(zp1, zp2)
istrat <- c(rep(1, length(zp1)), rep(2,length(zp2)))
ids <- !is.na(zp)
res <- concordancefit(y2[ids], zp[ids], istrat[ids], reverse=TRUE, timefix=FALSE)
cindex_te <- c("C"=res$concordance)#, "se(C)"=sqrt(res$var))

write.csv(cindex_te, '{output_r_path}')
"""

        with open(code_r_path, 'w') as ff:
            ff.write(rcode)
        #with open(os.devnull, 'w')  as FNULL:
        subprocess.check_call(['Rscript', code_r_path])#, stdout=FNULL)

        df_res = pd.read_csv(output_r_path)
        perf = df_res.iloc[0,1]
        #perf_se = df_res.iloc[1,1]
        perfs_cv.append(perf)

        os.remove(output_r_path)
        os.remove(dftr_r_path)
        os.remove(dfte_r_path)
        os.remove(code_r_path)

    perf = sum(perfs_cv) / len(perfs_cv)
    print(perf)

    res[outcomes.index(outcome), features.index(feature)] = perf

import pdb;pdb.set_trace()
df_res = pd.DataFrame(data=res, columns=features,index=outcomes)
df_res.to_csv(f'macrostructure_outcome_perf_future2.csv')


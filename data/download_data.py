import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm


base_dir = r'\\mgberisisilon1-mgmt.partners.org\PHS-RISC-LM4\ConvertedData\sleep\deidentified_sleep'
df = pd.read_csv('mastersheet_matched_Dementia.csv')
print(df.shape)

for i in tqdm(range(len(df))):
    folder = df.SignalPath.iloc[i].split('/')[-2]
    path = os.path.join(base_dir, folder)
    df.loc[i, 'exist'] = int(os.path.exists(path))
df = df[df.exist==1].reset_index(drop=True)
print(df.shape)

np.random.seed(2023)
ids = np.r_[np.random.choice(np.where(df.Y_Dementia==0)[0], 10),
            np.random.choice(np.where(df.Y_Dementia==1)[0], 10)]
df = df.iloc[ids].reset_index(drop=True)
cc = 0
for i in tqdm(range(len(df))):
    folder = df.SignalPath.iloc[i].split('/')[-2]
    path = os.path.join(base_dir, folder)
    if not os.path.exists(folder):
        shutil.copytree(path, folder)

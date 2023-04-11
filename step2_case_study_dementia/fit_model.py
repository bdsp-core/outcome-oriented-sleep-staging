from argparse import ArgumentParser
import os
import json
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch as th
from torch.utils.data import Subset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import sys
sys.path.insert(0, '../model_scripts')
from models import OOSSNet, OOSSDataset


def get_data(dataset, Ncv):
    # create dataloaders with cross-validation
    #cv_ids = np.arange(len(dataset))
    #np.random.shuffle(cv_ids)
    #cv_ids = np.array_split(cv_ids, Ncv)
    
    cvf = StratifiedKFold(n_splits=Ncv, shuffle=True)
    cv_ids = [teids for _, teids in cvf.split(np.zeros((len(dataset),1)), dataset.Y)]
    #cv_ids = [[i] for i in range(len(dataset))]
    
    ids_tr = []; ids_va = []; ids_te = []
    data_tr = []; data_va = []; data_te = []
    for cvi in range(len(cv_ids)):
        teids = cv_ids[cvi]
        
        not_te_foldids = np.setdiff1d(np.arange(Ncv), cvi)
        vaids = cv_ids[not_te_foldids[-1]]
        #not_te_ids = np.setdiff1d(np.arange(len(dataset)), teids)
        #cvf = StratifiedKFold(n_splits=Ncv, shuffle=True)
        #vaids = [vaid_ for _, vaid_ in cvf.split(np.zeros((len(not_te_ids),1)), dataset.Y[not_te_ids])][0]
        #vaids = not_te_ids[vaids]
        
        trids = np.setdiff1d(np.arange(len(dataset)), np.r_[vaids, teids])

        ids_tr.append( trids )
        ids_va.append( vaids )
        ids_te.append( teids )
        data_tr.append( Subset(dataset, trids) )
        data_va.append( Subset(dataset, vaids) )
        data_te.append( Subset(dataset, teids) )

    return data_tr, data_va, data_te, ids_tr, ids_va, ids_te


def main(data_path, Ncv=10, accelerator='auto', verbose=True, **hp):
    with open(data_path, 'rb') as ff:
        res = pickle.load(ff)
    dall = OOSSDataset(res)
    dtrs, dvas, dtes, ids_tr, ids_va, ids_te = get_data(dall, Ncv)

    models_cv = []
    sleep_stages_te = [None]*len(dall)
    for cvi, (dtr, dva, dte) in enumerate(zip(dtrs, dvas, dtes)):
        if verbose:
            print(f"""
===========================
Cross validation fold {cvi+1}/{len(dtrs)}
===========================
""")

        dtr_loader = DataLoader(dtr, batch_size=hp['batch_size'], shuffle=True, collate_fn=dall.collate_fn)
        dva_loader = DataLoader(dva, batch_size=len(dva), shuffle=False, collate_fn=dall.collate_fn)
        dte_loader = DataLoader(dte, batch_size=len(dte), shuffle=False, collate_fn=dall.collate_fn)

        model = OOSSNet(
                lr=hp['lr'], lr_reduce_patience=hp['lr_reduce_patience'],
                n_MCMC=hp['n_MCMC'],
                baseline_len=hp['baseline_len']
                )
                
        trainer = pl.Trainer(
                accelerator=accelerator,
                max_epochs=hp['max_epoch'],
                deterministic=True, log_every_n_steps=10,
                check_val_every_n_epoch=1,
                callbacks=[
                    ModelCheckpoint(save_top_k=1, monitor='val_loss', mode='min'),
                    EarlyStopping(monitor="val_loss", mode="min", patience=50),
                    LearningRateMonitor(logging_interval='epoch')],
                )
        trainer.fit(model, train_dataloaders=dtr_loader, val_dataloaders=dva_loader)
        #model = OOSSNet.load_from_checkpoint("/data/cdac Dropbox/a_People_BIDMC/Haoqi/outcome-oriented-sleep-staging-goodbye-aasm/github_repo/step2_case_study/lightning_logs/version_7/checkpoints/epoch=770-step=6168.ckpt")
        models_cv.append(model)
        
        for pn,p in model.named_parameters():
            print(f'{pn} = {p.data.cpu().numpy()}')
            
        #trainer.test(model, dataloaders=dte_loader, ckpt_path='best')
        ss = trainer.predict(model, dte_loader, ckpt_path='best')[0].numpy().astype(float)

        for j, idx in enumerate(ids_te[cvi]):
            sleep_stages_te[idx] = ss[j][~np.isnan(ss[j])]
    
    import pdb;pdb.set_trace()
    with open('sleep_stages_te.pickle', 'wb') as ff:
        pickle.dump(sleep_stages_te, ff)
        
    dall_loader = DataLoader(dall, batch_size=len(dall), shuffle=False, collate_fn=dall.collate_fn)
    thress = [float(th.sigmoid(m.qz_logit_thres).data.cpu().numpy()) for m in models_cv]
    idx = np.argsort(thress)[len(thress)//2]
    model = models_cv[idx]
    sleep_stages_newn2n3 = trainer.predict(model, dall_loader)[0].numpy().astype(float)
    sleep_stages_newn2n3 = [x[~np.isnan(x)] for x in sleep_stages_newn2n3]
    
    sleep_stages_new = []
    for i in range(len(dall)):
        ss_ = np.array(dall.S[i])
        ss_[dall.S[i]<=2] = (sleep_stages_newn2n3[i]<0.5).astype(int)+1
        sleep_stages_new.append(ss_)
    new_n3_perc = np.array([(x==1).mean() for x in sleep_stages_new])
    n3_perc = np.array([(x==1).mean() for x in dall.S])
    with open('sleep_stages_new.pickle', 'wb') as ff:
        pickle.dump(sleep_stages_new, ff)

if __name__=='__main__':
    parser = ArgumentParser(description='SWA percent for dementia')
    parser.add_argument('-i', '--datapath')
    parser.add_argument('-p', '--hyperparam', default='hyperparams.json')
    parser.add_argument('-k', '--checkpointpath', default='checkpoints')
    parser.add_argument('-c', '--Ncv', default=10)
    parser.add_argument('-s', '--seed', default=2023)
    parser.add_argument('-v', '--verbose', default=True)
    parser.add_argument('-a', '--accelerator', default='auto')
    args = parser.parse_args()

    with open(args.hyperparam, 'r') as f:
        hyperparam = json.load(f)
    pl.seed_everything(args.seed)
    main(
        args.datapath, Ncv=args.Ncv, 
        accelerator=args.accelerator, verbose=args.verbose,
        **hyperparam)


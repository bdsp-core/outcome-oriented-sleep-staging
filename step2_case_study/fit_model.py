from argparse import ArgumentParser
import os
import json
import numpy as np
import torch as th
from torch.utils.data import Subset, DataLoader
import lightning.pytorch as pl
import sys
sys.path.insert(0, '../model_scripts')
from models import OOSSNet, OOSSDataset


def get_data(dataset, Ncv):
    # create dataloaders with cross-validation
    cv_ids = np.arange(len(dataset))
    np.random.shuffle(cv_ids)
    cv_ids = np.array_split(cv_ids, Ncv)

    data_tr = []; data_va = []; data_te = []
    for cvi in range(Ncv):
        teids = cv_ids[cvi]
        not_te_foldids = np.setdiff1d(np.arange(Ncv), cvi)
        vaids = cv_ids[not_te_foldids[-1]]
        trids = np.setdiff1d(np.arange(len(dataset)), np.r_[vaids, teids])

        data_tr.append( Subset(dataset, trids) )
        data_va.append( Subset(dataset, vaids) )
        data_te.append( Subset(dataset, teids) )

    return data_tr, data_va, data_te


def main(data_path, Ncv=5, use_gpu=False, verbose=True, **hp):#, ckpt_path='checkpoints'
    #os.makedirs(ckpt_path, exist_ok=True)
    dall = OOSSDataset(data_path)
    dtrs, dvas, dtes = get_data(dall, Ncv)

    sleep_stages_new = []
    for cvi, (dtr, dva, dte) in enumerate(zip(dtrs, dvas, dtes)):
        if verbose:
            print(f"""
=========================
Cross validation fold {cvi+1}/{Ncv}
=========================
""")

        dtr_loader = DataLoader(dtr, batch_size=hp['batch_size'], shuffle=True, collate_fn=dall.collate_fn)
        dva_loader = DataLoader(dva, batch_size=hp['batch_size'], shuffle=False, collate_fn=dall.collate_fn)
        dte_loader = DataLoader(dte, batch_size=hp['batch_size'], shuffle=False, collate_fn=dall.collate_fn)

        model = OOSSNet(
                lr=hp['lr'], n_MCMC=hp['n_MCMC'],
                baseline_len=hp['baseline_len']
                )
        trainer = pl.Trainer(
                accelerator='gpu' if use_gpu else 'cpu',
                max_epochs=hp['max_epoch'],
                deterministic=True, log_every_n_steps=5,
                )
        trainer.fit(model, train_dataloaders=dtr_loader, val_dataloaders=dva_loader)#, ckpt_path=ckpt_path)
        import pdb;pdb.set_trace()

        trainer.test(model, dataloaders=dte_loader, ckpt_path='best')
        sleep_stages_new.append( trainer.predict(model, dte_loader, ckpt_path='best') )


if __name__=='__main__':
    parser = ArgumentParser(description='SWA percent for dementia')
    parser.add_argument('-d', '--datapath')
    parser.add_argument('-p', '--hyperparam', default='hyperparams.json')
    parser.add_argument('-k', '--checkpointpath', default='checkpoints')
    parser.add_argument('-c', '--Ncv', default=5)
    parser.add_argument('-s', '--seed', default=2023)
    parser.add_argument('-v', '--verbose', default=True)
    parser.add_argument('-g', '--gpu', action='store_true')
    args = parser.parse_args()

    with open(args.hyperparam, 'r') as f:
        hyperparam = json.load(f)
    pl.seed_everything(args.seed)
    main(
        args.datapath, Ncv=args.Ncv, 
        use_gpu=args.gpu, verbose=args.verbose,
        #ckpt_path=args.checkpointpath,
        **hyperparam)


from argparse import ArgumentParser
import numpy as np
import sys
sys.path.insert(0, '../model_scripts')
from sklearn.model_selection import KFold
import torch as th
th.backends.cudnn.determinstic = True
th.backends.cudnn.benchmark = False
from models import OOSSNet, OOSSDataset
import pytorch_lightning as pl
from lightning import Trainer


def get_data(dataset, Ncv):
    # create dataloaders with cross-validation
    cv_ids = np.arange(len(dataset))
    np.random.shuffle(cv_ids)
    cv_ids = np.array_split(cv_ids, Ncv)

    data_tr = []; data_va = []; data_te = []
    for cvi in range(Ncv):
        teids = cv_ids[cvi]
        not_teids = np.setdiff1d(np.arange(Ncv), cvi)
        vaids = cv_ids[not_teids[-1]]
        trids = np.setdiff1d(np.arange(len(dataset)), np.r_[vaids, teids])

        dtr = Subset(dataset, trids)
        dva = Subset(dataset, vaids)
        dte = Subset(dataset, teids)

        data_tr.append(DataLoader(dtr))
        data_va.append(DataLoader(dva))
        data_te.append(DataLoader(dte))
    return data_tr, data_va, data_te


def main(data_path, Ncv=5, use_gpu=False, verbose=True):
    dall = OOSSDataset(data_path)
    dtrs, dvas, dtes = get_data(dall, Ncv)

    for cvi, (dtr, dva, dte) in enumerate(zip(dtrs, dvas, dtes)):
        if verbose:
            print(f'\n========\nCross validation fold {cvi+1}/{Ncv}\n========\n')

        model = OOSSNet()
        trainer = Trainer()
        trainer.fit(model, dtr, dva, dte)


if __name__=='__main__':
    parser = ArgumentParser(description='SWA percent for dementia')
    parser.add_argument('-d', '--datapath')
    parser.add_argument('-c', '--Ncv', default=5)
    parser.add_argument('-s', '--seed', default=2023)
    parser.add_argument('-v', '--verbose', default=True)
    parser.add_argument('-g', '--gpu', action='store_true')
    args = parser.parse_args()
    import pdb;pdb.set_trace()
    pl.seed_everything(args.seed)

    main(args.datapath, Ncv=args.Ncv, use_gpu=args.gpu, verbose=args.verbose)


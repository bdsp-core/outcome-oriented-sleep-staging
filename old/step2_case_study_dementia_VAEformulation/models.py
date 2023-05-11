from collections import deque
import numpy as np
from sklearn.metrics import roc_auc_score
import torch as th
from torch import nn
from torch.utils.data import Dataset
from torch.distributions import Normal, Bernoulli
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning.pytorch as pl


zero_eps = 1e-6
one_eps  = 1-1e-6

class OOSSDataset(Dataset):
    """
    sid: subject id
    X: signal or feature
    S: AASM sleep stage
    Y: outcome
    L: covariates
    """
    def __init__(self, data_pickle):
        self.X = data_pickle['X']; self.Xnames = data_pickle['Xnames']
        self.S = data_pickle['S']
        self.Y = data_pickle['Y']
        self.T = np.array([len(x) for x in self.S])
        #self.L = data_pickle['L']; self.Lnames = data_pickle['Lnames']
        #self.sids = data_pickle['sids']
        self.var_names = ['X', 'S', 'Y', 'T']#, 'L'
        self.var2difflen = {'X':True, 'S':True, 'Y':False, 'T':False}
        self.var2type = {'X':'torch.FloatTensor', 'S':'torch.FloatTensor', 'Y':'torch.IntTensor', 'T':'torch.FloatTensor'}

        # only take N2 and N3
        self.X = [x[s<=2] for x,s in zip(self.X, self.S)]

        # squeeze [0,1] to (0,1)
        for i in range(len(self.X)):
            self.X[i][self.X[i]==0] = zero_eps
            self.X[i][self.X[i]==1] = one_eps

    def __getitem__(self, idx):
        return {x:getattr(self, x)[idx] for x in self.var_names}

    def __len__(self):
        return len(self.T)

    def collate_fn(self, batch):
        res = {}
        for x in self.var_names:
            tp = self.var2type[x]
            if type(batch)==list:
                if self.var2difflen[x]:
                    # pad to form masked tensor
                    res_raw = [y[x] for y in batch]
                    maxT = max([len(y) for y in res_raw])
                    if res_raw[0].ndim==2:
                        res[x] = th.tensor(np.array([np.pad(y, ((0,maxT-len(y)),(0,0)), constant_values=0.5) for y in res_raw])).type(tp)
                    else:
                        res[x] = th.tensor(np.array([np.pad(y, ((0,maxT-len(y)),), constant_values=0.5) for y in res_raw])).type(tp)
                    res[x+'_mask'] = th.tensor(np.array([np.pad(np.ones(len(y)), ((0,maxT-len(y)),)) for y in res_raw])).float()
                    #res[x] = tensor(val, mask)
                else:
                    res[x] = th.tensor([y[x] for y in batch]).type(tp)
            else:
                res[x] = th.tensor(batch[x]).type(tp)
        return res


class OOSSNet(pl.LightningModule):
    """
    """
    def __init__(self, lr=1e-3, lr_reduce_patience=10, n_MCMC=100, baseline_len=0):
        super().__init__()
        self.save_hyperparameters()
        self.previous_f = deque(maxlen=self.hparams.baseline_len)
        scale = 0.1
        
        # q(Z|S,X,L)
        self.qz_logit_thres = nn.Parameter(th.randn(())*scale)
        self.qz_log_slope = nn.Parameter(th.randn(())*scale)

        # p(X|Z)
        self.px_mus0 = nn.Parameter(th.randn((1,))*scale)
        self.px_log_mus1_delta = nn.Parameter(th.randn((1,))*scale)
        self.px_log_sigmas = nn.Parameter(th.randn((2,))*scale)

        # p(Z|Y,L)
        self.pz_mus0 = nn.Parameter(th.randn((1,))*scale)
        self.pz_log_mus1_delta = nn.Parameter(th.randn((1,))*scale)
        self.pz_log_sigmas = nn.Parameter(th.randn((2,))*scale)
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=self.hparams.lr_reduce_patience)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_loss",
            },
        }

    def training_step(self, batch, batch_idx):
        return self._forward(batch, batch_idx, 'train')
    def validation_step(self, batch, batch_idx):
        self._forward(batch, batch_idx, 'val')
    def test_step(self, batch, batch_idx):
        self._forward(batch, batch_idx, 'test')
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        qz = self(batch)
        for i in range(len(qz)):
            qz[i, batch['X_mask'][i].sum().int():] = np.nan
        return qz

    def forward(self, batch, sigmoid=True):
        # q(Z|S,X,L)
        thres = th.sigmoid(self.qz_logit_thres)
        slope = th.exp(self.qz_log_slope)
        logit_qz = slope * (batch['X'][...,0] - thres)#TODO remove [,0]
        if sigmoid:
            return th.sigmoid(logit_qz)
        else:
            return logit_qz

    def _forward(self, batch, batch_idx, prefix):
        X, Xmask, Y, T = batch['X'], batch['X_mask'], batch['Y'], batch['T']

        # encode: q(Z|S,X,L)
        logit_qz = self(batch, sigmoid=False)

        # sample z from log_pz
        z_dist_q = Bernoulli(logits=logit_qz)
        z_mcmc = z_dist_q.sample((self.hparams.n_MCMC,)).data
        log_qz = z_dist_q.log_prob(z_mcmc)
        log_qz = (log_qz*Xmask).mean(dim=-1)

        # decode: p(X|Z)
        px_mus = th.cat([self.px_mus0, self.px_mus0+th.exp(self.px_log_mus1_delta)])
        px_sigmas  = th.exp(self.px_log_sigmas)
        x_dist = Normal(
            px_mus.index_select(0, z_mcmc.int().flatten()).reshape(z_mcmc.shape),
            px_sigmas.index_select(0, z_mcmc.int().flatten()).reshape(z_mcmc.shape))
        log_px = x_dist.log_prob(th.logit(X[...,0]))#TODO
        log_px = (log_px*Xmask).mean(dim=-1)

        # regularizer: p(Z|Y,L)
        pz_mus = th.cat([self.pz_mus0, self.pz_mus0-th.exp(self.pz_log_mus1_delta)])
        pz_sigmas  = th.exp(self.pz_log_sigmas)
        z_dist_p = Normal(pz_mus.index_select(0,Y), pz_sigmas.index_select(0,Y))
        z2 = (z_mcmc*Xmask).sum(dim=-1)/T
        z2 = th.clamp(z2, zero_eps, one_eps)
        log_pz = z_dist_p.log_prob(th.logit(z2))
        
        fs = log_px + log_pz*10 - log_qz

        # get loss
        if len(self.previous_f)==0:
            baseline = 0
        else:
            baseline = th.Tensor(self.previous_f).mean()
        loss = -th.mean(log_qz*(fs.data - baseline) + fs)
        if prefix=='train':
            self.previous_f.append(fs.data.mean())

        yp = (((logit_qz.data>0).float()*Xmask).sum(dim=-1)/T).cpu().numpy().astype(float)
        y = Y.cpu().numpy().astype(int)
        auc = roc_auc_score(y, yp)

        self.log_dict({
            prefix+'_loss': loss.data,
            prefix+'_rec_log_px':log_px.data.mean(),
            prefix+'_log_pz':log_pz.data.mean(),
            prefix+'_log_qz':log_qz.data.mean(),
            prefix+'_log_pz_minus_log_qz':log_pz.data.mean()-log_qz.data.mean(),
            prefix+'_f':fs.data.mean(),
            prefix+'_metric':auc,
            prefix+'_metric2':max(auc,1-auc),
        })

        return loss

    def on_train_start(self):
        self.previous_f.clear()
    def on_validation_start(self):
        self.previous_f.clear()
    def on_test_start(self):
        self.previous_f.clear()
    def on_predict_start(self):
        self.previous_f.clear()
    def on_fit_start(self):
        self.previous_f.clear()


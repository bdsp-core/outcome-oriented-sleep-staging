from collections import deque
import pickle
import numpy as np
import torch as th
from torch import nn
from torch.utils.data import Dataset
from torch.distributions import Beta, Bernoulli
from torch.optim import Adam
import lightning.pytorch as pl


class OOSSDataset(Dataset):
    """
    sid: subject id
    X: signal or feature
    S: AASM sleep stage
    Y: outcome
    L: covariates
    """
    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as ff:
            res = pickle.load(ff)
        self.X = res['X']; self.Xnames = res['Xnames']
        self.Y = res['Y']
        self.T = np.array([len(x) for x in res['S']])
        #self.S = res['S']
        #self.L = res['L']; self.Lnames = res['Lnames']
        #self.sids = res['sids']
        self.var_names = ['X', 'Y', 'T']#, 'S', 'L'
        self.var2difflen = {'X':True, 'Y':False, 'T':False}
        self.var2type = {'X':'torch.FloatTensor', 'Y':'torch.IntTensor', 'T':'torch.FloatTensor'}

        # only take N2 and N3
        self.X = [x[s<=2] for x,s in zip(self.X, res['S'])]

        # squeeze [0,1] to (0,1)
        N = self.T.sum()
        for i in range(len(self.X)):
            self.X[i][self.X[i]==0] = 0.5/N
            self.X[i][self.X[i]==1] = (N-0.5)/N

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
                    res[x] = th.tensor(np.array([np.pad(y, ((0,maxT-len(y)),(0,0)), constant_values=0.5) for y in res_raw])).type(tp)
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
    def __init__(self, lr=1e-3, n_MCMC=100, baseline_len=0):
        super().__init__()
        self.save_hyperparameters()
        self.previous_f = deque(maxlen=self.hparams.baseline_len)

        # q(Z|S,X,L)
        self.qz_logit_thres = nn.Parameter(th.randn(()))
        self.qz_log_slope = nn.Parameter(th.randn(()))

        # p(X|Z)
        self.px_log_alpha = nn.Parameter(th.randn((2,)))
        self.px_log_beta = nn.Parameter(th.randn((2,)))

        # p(Z|Y,L)
        self.pz_log_alpha = nn.Parameter(th.randn((2,)))
        self.pz_log_beta = nn.Parameter(th.randn((2,)))

    def training_step(self, batch, batch_idx):
        return self._forward(batch, batch_idx, 'train')
    def validation_step(self, batch, batch_idx):
        self._forward(batch, batch_idx, 'validation')
    def test_step(self, batch, batch_idx):
        self._forward(batch, batch_idx, 'test')
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, x, sigmoid=True):
        # q(Z|S,X,L)
        thres = th.sigmoid(self.qz_logit_thres)
        slope = th.exp(self.qz_log_slope)
        logit_pz = slope * (x[...,0] - thres)#TODO remove [,0]
        if sigmoid:
            return th.sigmoid(logit_pz)
        else:
            return logit_pz

    def _forward(self, batch, batch_idx, prefix):
        X, Xmask, Y, T = batch['X'], batch['X_mask'], batch['Y'], batch['T']

        # encode: q(Z|S,X,L)
        logit_qz = self(X, sigmoid=False)

        # sample z from log_pz
        px_alpha = th.exp(self.px_log_alpha)
        px_beta  = th.exp(self.px_log_beta)
        z_dist_q = Bernoulli(logits=logit_qz)
        z_mcmc = z_dist_q.sample((self.hparams.n_MCMC,)).data
        log_qz = z_dist_q.log_prob(z_mcmc)
        log_qz = (log_qz*Xmask).sum(dim=-1)

        # decode: p(X|Z)
        x_dist = Beta(
            px_alpha.index_select(0, z_mcmc.int().flatten()).reshape(z_mcmc.shape),
            px_beta.index_select(0, z_mcmc.int().flatten()).reshape(z_mcmc.shape))
        log_px = x_dist.log_prob(X[...,0])#TODO
        log_px = (log_px*Xmask).sum(dim=-1)

        # regularizer: p(Z|Y,L)
        pz_alpha = th.exp(self.pz_log_alpha)
        pz_beta  = th.exp(self.pz_log_beta)
        z_dist_p = Beta(pz_alpha.index_select(0,Y), pz_beta.index_select(0,Y))
        log_pz = z_dist_p.log_prob((z_mcmc*Xmask).sum(dim=-1)/T)
        
        fs = log_px + log_pz - log_qz

        # get loss
        if len(self.previous_f)==0:
            baseline = 0
        else:
            baseline = th.Tensor(self.previous_f).mean()
        loss = -th.mean(log_qz*(fs.data - baseline) + fs)
        if prefix=='train':
            self.previous_f.append(fs.data.mean())

        self.log_dict({
            prefix+'_loss': loss,
            prefix+'_rec_log_px':log_px.mean(),
            prefix+'_D_kl_qz_pz':log_qz.mean()-log_pz.mean(),
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


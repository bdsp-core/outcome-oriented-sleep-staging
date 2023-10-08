import logging
import os
import shutil
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy.special import expit as sigmoid
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import lightning
lightning.fabric.utilities.seed.seed_everything(2023)


class MyDataset(Dataset):
    def __init__(self, X, S, y=None):
        super().__init__()
        self.X = X
        self.S = S
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return (self.X[idx], self.S[idx])
        else:
            return (self.X[idx], self.S[idx], self.y[idx])
    
    @classmethod
    def collate(cls, batch, split_to_shorter=False):
        has_y = type(batch[0])==tuple
        if split_to_shorter:
            batch_x = []
            batch_s = []
            batch_y = []
            for v in batch:
                xx = np.array_split(v[0], 5, axis=0)
                batch_x.extend(xx)
                ss = np.array_split(v[1], 5, axis=0)
                batch_s.extend(ss)
                if has_y:
                    batch_y.extend([v[2]]*len(xx))
        else:
            batch_x = [v[0] for v in batch]
            batch_s = [v[1] for v in batch]
            if has_y:
                batch_y = [v[2] for v in batch]
        T = [len(x) for x in batch_x]
        maxT = max(T)
        x_pr = np.array([np.pad(x, ((0,maxT-len(x)), (0,0))) for x in batch_x])  # pad right
        x_pl = np.array([np.pad(x, ((maxT-len(x),0), (0,0))) for x in batch_x])  # pad left
        s_pr = np.array([np.pad(x, ((0,maxT-len(x)), (0,0))) for x in batch_s])  # pad right
        s_pl = np.array([np.pad(x, ((maxT-len(x),0), (0,0))) for x in batch_s])  # pad left
        res = (
            th.tensor(x_pr).float(), th.tensor(x_pl).float(),
            th.tensor(s_pr).float(), th.tensor(s_pl).float(),
            th.tensor(T).long())
        if has_y:
            res = res + (th.tensor(batch_y),)
        return res
        

def th2np(x, dtype=float):
    assert type(x) in [th.Tensor, nn.Parameter]
    return np.array(x.data.cpu().tolist()).astype(dtype)


def log_matmul(log_A, log_B):
	"""
	log_A : m x n
	log_B : n x p
	output : m x p matrix

	Normally, a matrix multiplication
	computes out_{i,j} = sum_k A_{i,k} x B_{k,j}

	A log domain matrix multiplication
	computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}
	"""
	m = log_A.shape[0]
	n = log_A.shape[1]
	p = log_B.shape[1]

	log_A_expanded = th.reshape(log_A, (m,n,1))
	log_B_expanded = th.reshape(log_B, (1,n,p))

	elementwise_sum = log_A_expanded + log_B_expanded
	out = th.logsumexp(elementwise_sum, dim=1)

	return out


#class TransformerOOSClassifier
class HMMOOSSClassifier(BaseEstimator, ClassifierMixin, LightningModule):
    def __init__(self, Xnames, bin_mask, n_components=5, class_weight=None, C_l1=0.01, C_Y=1, C_emission=0.01, lr=0.001, max_iter=100, batch_size=8, lr_reduce_patience=3, early_stop_patience=10, log_dir=None, random_state=None, verbose=True, warmstart_model_folder=''):#TODO have a separate emission_model
        super().__init__()
        self.save_hyperparameters()
        self.Xnames = Xnames
        self.bin_mask = bin_mask
        self.n_components = n_components
        self.class_weight = class_weight
        self.C_l1 = C_l1
        self.C_Y = C_Y
        self.C_emission = C_emission
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr_reduce_patience = lr_reduce_patience
        self.early_stop_patience = early_stop_patience
        self.random_state = random_state
        self.log_dir = log_dir
        self.verbose = verbose
        self.warmstart_model_folder = warmstart_model_folder
        self.validation_outputs = []
        
        if verbose:
            logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.INFO)
            logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.INFO)
        else:
            logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
            logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

    def get_emission_logP(self, X, S):
        """
        get log P(observation|Z),
        shape = (#state, batch_size, max T, #feature)
        """
        # get log P(X_1:T|Z)
        X_cont = X[...,(~self.bin_mask).tolist()]  # shape=(B,T,D)
        X_cont = X_cont[None,...] # shape=(1,B,T,D)
        mean_ = self.emission_X_gaussian_mean[:,None,None,:] # shape=(NC,1,1,D)
        sd_log_ = self.emission_X_gaussian_sd_log[:,None,None,:] # shape=(NC,1,1,D)
        logP_X_Z_cont = -sd_log_-((X_cont-mean_)/th.exp(sd_log_))**2/2  # -np.log(np.sqrt(2*np.pi))
        X_bin = X[...,self.bin_mask.tolist()]  # shape=(B,T,D)
        X_bin = X_bin[None,...] # shape=(1,B,T,D)
        logit_p_ = self.emission_X_bernoulli_logit[:,None,None,:] # shape=(NC,1,1,D)
        logP_X_Z_bin = F.logsigmoid(logit_p_)*X_bin + F.logsigmoid(-logit_p_)*(1-X_bin)
        logP_X_Z = logP_X_Z_cont.sum(dim=-1)+logP_X_Z_bin.sum(dim=-1)

        # get log P(S_1:T|Z)
        log_p_ = F.log_softmax(F.pad(self.emission_S_unnormalized,(1,0,0,0), 'constant',0), dim=1)  # shape=(NC,3)
        log_p_ = log_p_[:,None,None,:] # shape=(NC,1,1,3)
        S_ = S[None,...] # shape=(1,B,T,3)
        logP_S_Z = (log_p_*S_).sum(dim=-1)

        # get log P(X_1:T, S_1:T|Z)
        return logP_X_Z + logP_S_Z
    
    def _hmm_forward(self, X, S, T, log_transition_matrix, log_state_prior):
        """
        forward pass:
        compute likelihood P(X_1:T,S_1:T) and alpha
        """
        N, Tmax, D = X.shape
        logP_XS_Z = self.get_emission_logP(X, S)

        ## forward pass -- run
        # c_t alpha(Zt) = P(Xt|Zt)P(St|Zt)\sum_z_t-1 alpha(z_t-1)p(Zt|zt-1)
        # c_1 = P(X1,S1) = \sum_z P(X1|z)P(S1|z)p(z)
        # alpha(Z1) = P(Z1|X1,S1) = P(X1|Z1)P(S1|Z1)P(Z1)/c_1
        log_c = []
        log_alpha = []
        for t in range(Tmax):
            if t==0:
                tmp = logP_XS_Z[...,t]+log_state_prior[...,None]
            else:
                tmp = logP_XS_Z[...,t]+log_matmul(log_transition_matrix.t(), log_alpha[-1])
            log_c.append( th.logsumexp(tmp,dim=0) )
            log_alpha.append( tmp-log_c[-1] )

        # sum(c_1:T) is the log-likelihood (each sample may have different length).
        log_c = th.stack(log_c) # shape=(Tmax,B)
        ll = th.stack([log_c[:T[i],i].mean() for i in range(N)]) # .mean to bring down its scale
        
        log_alpha = th.stack(log_alpha) # shape=(Tmax,K,B)

        return ll, log_alpha, log_c
        
    def _hmm_backward(self, X, S, log_c, log_transition_matrix, log_state_prior):
        """
        backward pass
        c_t+1 beta(Zt) = \sum_z_t+1 beta(zt+1)P(Xt+1|zt+1)P(St+1|zt+1)p(zt+1|Zt)
        beta(zN) = 1
        """
        N, Tmax, D = X.shape
        logP_XS_Z = self.get_emission_logP(X, S)

        #log_c = []
        log_beta = [ th.zeros(self.n_components,N) ]
        for t in range(Tmax-2,-1,-1):
            if t==Tmax-2:
                tmp = log_matmul(log_transition_matrix, logP_XS_Z[...,t+1])
            else:
                tmp = log_matmul(log_transition_matrix, log_beta[0]+logP_XS_Z[...,t+1])
            #log_c.insert(0, th.logsumexp(tmp,dim=0))
            #if t>=0:
            log_beta.insert(0, tmp-log_c[t+1])
        #log_c = th.stack(log_c) # shape=(Tmax,B)
        #ll = th.stack([log_c[-T[i]:,i].mean() for i in range(N)])

        log_beta = th.stack(log_beta) # shape=(Tmax,K,B)

        return log_beta
        
    def forward(self, Xr, Xl, Sr, Sl, T, do_backward=True):
        """
        get log-likelihood, alpha from forward pass,
        and optionally beta from backward pass to get P(Z|X,S)
        """
        N = len(T)
        log_transition_matrix = F.log_softmax(F.pad(self.transition_matrix_unnormalized, (1,0,0,0), "constant", 0), dim=1)
        log_state_prior = F.log_softmax(F.pad(self.state_prior_unnormalized, (1,0), 'constant', 0), dim=0)

        ll, log_alpha, log_c = self._hmm_forward(Xr, Sr, T, log_transition_matrix, log_state_prior)

        res = [ll]
        if do_backward:
            # roll log_c from right-pad to left-pad
            shifts = T.max()-T
            indices = (th.arange(log_c.shape[0])[:, None] - shifts[None, :]) % log_c.shape[0]
            log_c = th.gather(log_c, 0, indices)
            log_beta = self._hmm_backward(Xl, Sl, log_c, log_transition_matrix, log_state_prior)
            
            # compute posterior P(Z_t|X_1:T, S_1:T)
            zp = [th.exp(log_alpha[:T[i],:,i] + log_beta[-T[i]:,:,i]) for i in range(N)] # element shape=(T,K)

            zp_hist = th.vstack([z.sum(dim=0) for z in zp])
            #zp_hist = zp_hist[:,:-1]

            #TODO add flattened transition matrix to prediction of H
            H = th.matmul(zp_hist, self.coef_th_)+self.intercept_th_
            res.append(H)
            res.append(zp)
        if len(res)==1:
            res = res[0]
        return res
        
    def _loss_hmm(self, ll_hmm):
        loss = -ll_hmm.mean()
        # make sure emission_S is (smoothly) constrained to 0 or 1
        penalty = 100*(th.mean(self.emission_S_unnormalized**2)-10000)**2
        return loss+penalty, loss, penalty
        
    def _loss_Y(self, H, Y):
        loss_Y = []
        for d in range(self.Dy):
            loss_Y.append( (-F.logsigmoid(H[:,d][Y[:,d]==1]).sum()-F.logsigmoid(-H[:,d][Y[:,d]==0]).sum())/len(Y) )
        loss_Y = sum(loss_Y)/self.Dy
        reg_l1 = th.mean(th.abs(self.coef_th_))
        return loss_Y + reg_l1*self.C_l1, loss_Y, reg_l1*self.C_l1
    
    def _loss(self, H, Y, ll_hmm):
        loss_hmm_total, loss_hmm, s_penalty = self._loss_hmm(ll_hmm) 
        _, loss_Y, reg_l1 = self._loss_Y(H,Y)
        return loss_hmm_total + loss_Y*self.C_Y  + reg_l1*self.C_l1, loss_hmm, loss_Y*self.C_Y, reg_l1*self.C_l1, s_penalty
        
    def training_step(self, batch, batch_idx):
        Xr, Xl, Sr, Sl, T, Y = batch
        ll_hmm, H, zp = self(Xr, Xl, Sr, Sl, T)
        loss, loss_hmm, loss_Y, reg_l1, s_penalty = self._loss(H, Y, ll_hmm)
        self.log("tr_loss", loss)#, prog_bar=True)
        self.log("tr_loss_hmm", loss_hmm)#, prog_bar=True)
        self.log("tr_loss_Y", loss_Y)#, prog_bar=True)
        self.log("tr_reg_l1", reg_l1)
        self.log("tr_s_penalty", s_penalty)
        return loss
        
    def validation_step(self, batch, batch_idx):
        Xr, Xl, Sr, Sl, T, Y = batch
        ll_hmm, H, zp = self(Xr, Xl, Sr, Sl, T)
        loss, loss_hmm, loss_Y, reg_l1, s_penalty = self._loss(H, Y, ll_hmm)
        self.log("val_loss", loss)#, prog_bar=True)
        self.log("val_loss_hmm", loss_hmm)#, prog_bar=True)
        self.log("val_loss_Y", loss_Y)#, prog_bar=True)
        res = {'H':H, 'zp':zp, 'Y':Y}
        self.validation_outputs.append(res)
        return res
        
    def predict_step(self, batch, batch_idx):
        Xr, Xl, Sr, Sl, T = batch
        _, H, zp = self(Xr, Xl, Sr, Sl, T)
        H = H.cpu().numpy().astype(float)
        zp = [z.cpu().numpy().astype(float) for z in zp]
        return {'H':H, 'zp':zp}
        
    def configure_optimizers(self):
        #TODO Levenberg–Marquardt algorithm?
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=self.lr_reduce_patience, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=1e-7, verbose=self.verbose)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_loss",
            },
        }
        
    def on_validation_epoch_end(self):
        #if len(self.validation_outputs)==0:
        H = np.concatenate([th2np(x['H']) for x in self.validation_outputs]).astype(float)
        Y = np.concatenate([th2np(x['Y']) for x in self.validation_outputs]).astype(float)
        self.val_output_yp = sigmoid(H)
        self.val_output_y = Y
        self.validation_outputs.clear()
    
    def _get_trainer(self):
        return Trainer(
                logger = TensorBoardLogger(save_dir=self.log_dir),
                accelerator='auto', max_epochs=self.max_iter,
                #deterministic=True,
                check_val_every_n_epoch=1, log_every_n_steps=3,
                callbacks=[
                    ModelCheckpoint(save_top_k=1, monitor='val_loss', mode='min'),
                    EarlyStopping(monitor="val_loss", mode="min", patience=self.early_stop_patience),# min_delta=self.tol),
                    LearningRateMonitor(logging_interval='epoch'),
                    ],
                    enable_progress_bar=self.verbose,
                    enable_model_summary=self.verbose,
                )
                
    def fit(self, XS, Y, sample_weight=None, separate=False):
        path = os.path.join(self.warmstart_model_folder, f'n_state={self.n_components}.pth')
        self.Dy = Y.shape[1]
        if os.path.exists(path):
            weights = th.load(path)
            self.state_prior_unnormalized = nn.Parameter(weights['state_prior_unnormalized'])
            self.transition_matrix_unnormalized = nn.Parameter(weights['transition_matrix_unnormalized'])
            self.emission_X_bernoulli_logit = nn.Parameter(weights['emission_X_bernoulli_logit'])
            self.emission_X_gaussian_mean = nn.Parameter(weights['emission_X_gaussian_mean'])
            self.emission_X_gaussian_sd_log = nn.Parameter(weights['emission_X_gaussian_sd_log'])
            self.emission_S_unnormalized = nn.Parameter(weights['emission_S_unnormalized'])
            self.coef_th_ = nn.Parameter(weights['coef_th_'])
            self.intercept_th_ = nn.Parameter(weights['intercept_th_'])
            print(f'warm start weights loaded from {path}!')
        else:
            NC = self.n_components
            D = len(self.Xnames)
            scale = 1/np.sqrt(D)
            tm = th.eye(NC)*2+th.randn(NC,NC)*scale
            tm = tm-tm[:,[0]]
            self.state_prior_unnormalized = nn.Parameter(th.randn(NC-1)*scale)
            self.transition_matrix_unnormalized = nn.Parameter(tm[:,1:])
            self.emission_X_bernoulli_logit = nn.Parameter(th.randn(NC,self.bin_mask.sum())*scale)
            self.emission_X_gaussian_mean = nn.Parameter(th.randn(NC,(~self.bin_mask).sum())*scale)
            self.emission_X_gaussian_sd_log = nn.Parameter(th.randn(NC,(~self.bin_mask).sum())*scale)
            v = th.randn(NC,3-1) # 3 is W,NREM,REM
            v = v/th.sqrt(th.mean(v**2))*100 # contrain mean(v**2) to be big
            self.emission_S_unnormalized = nn.Parameter(v)
            self.coef_th_ = nn.Parameter(th.randn(NC,self.Dy)*scale)
            self.intercept_th_ = nn.Parameter(th.tensor(np.zeros(self.Dy)))
            
        X = [x[0] for x in XS]
        S = [x[1] for x in XS]
        N = len(Y)
        assert len(X)==N, f'len(X)={len(X)}!=len(Y)={N}'
        vaids = np.sort(np.random.choice(N, N//5, replace=False))
        trids = np.setdiff1d(np.arange(N), vaids)
        
        # standardize
        X2 = np.concatenate(X, axis=0)[:,~self.bin_mask]
        self.Xmean_ = np.zeros(len(self.Xnames))
        self.Xmean_[~self.bin_mask] = np.nanmean(X2, axis=0)
        self.Xscale_ = np.ones(len(self.Xnames))
        self.Xscale_[~self.bin_mask] = np.nanstd(X2, axis=0)
        X = [(x-self.Xmean_)/self.Xscale_ for x in X]

        #TODO deal with class_weight and sample_weight, especially Y is multiple targets
        
        dataset = MyDataset(X, S, Y)
        dtr = Subset(dataset, trids)
        loader_tr = DataLoader(dtr, batch_size=self.batch_size, num_workers=0, shuffle=True, collate_fn=MyDataset.collate)
        dva = Subset(dataset, vaids)
        loader_va = DataLoader(dva, batch_size=self.batch_size, num_workers=0, shuffle=False, collate_fn=MyDataset.collate)
        
        if separate:
            old_training_step = self.training_step
            old_validation_step = self.validation_step
            old_on_validation_epoch_end = self.on_validation_epoch_end
            
            #collate_fn2 = lambda batch: MyDataset.collate(batch, split_to_shorter=True)
            #loader_tr.collate_fn = collate_fn2
            #loader_va.collate_fn = collate_fn2
            self.coef_th_.requires_grad = False
            self.intercept_th_.requires_grad = False
            def training_step(batch, batch_idx):
                Xr, Xl, Sr, Sl, T, Y = batch
                ll_hmm = self(Xr, Xl, Sr, Sl, T, do_backward=False)
                loss_total, loss, s_penalty  = self._loss_hmm(ll_hmm) 
                self.log("tr_loss_total", loss_total)#, prog_bar=True)
                self.log("tr_loss", loss)#, prog_bar=True)
                self.log("tr_s_penalty", s_penalty)#, prog_bar=True)
                return loss
            def validation_step(batch, batch_idx):
                Xr, Xl, Sr, Sl, T, Y = batch
                ll_hmm = self(Xr, Xl, Sr, Sl, T, do_backward=False)
                loss_total, loss, s_penalty  = self._loss_hmm(ll_hmm) 
                self.log("val_loss_total", loss_total)#, prog_bar=True)
                self.log("val_loss", loss)#, prog_bar=True)
                self.log("val_s_penalty", s_penalty)#, prog_bar=True)
            def on_validation_epoch_end():
                pass
            self.training_step = training_step
            self.validation_step = validation_step
            self.on_validation_epoch_end = on_validation_epoch_end
            mytrainer1 = self._get_trainer()
            mytrainer1.fit(model=self, train_dataloaders=loader_tr, val_dataloaders=loader_va)
            
            #loader_tr.collate_fn = MyDataset.collate
            #loader_va.collate_fn = MyDataset.collate
            self.state_prior_unnormalized.requires_grad = False
            self.transition_matrix_unnormalized.requires_grad = False
            self.emission_X_bernoulli_logit.requires_grad = False
            self.emission_X_gaussian_mean.requires_grad = False
            self.emission_X_gaussian_sd_log.requires_grad = False
            self.emission_S_unnormalized.requires_grad = False
            self.coef_th_.requires_grad = True
            self.intercept_th_.requires_grad = True
            def training_step(batch, batch_idx):
                Xr, Xl, Sr, Sl, T, Y = batch
                _, H, zp = self(Xr, Xl, Sr, Sl, T)
                loss, loss_Y, reg_l1 = self._loss_Y(H,Y)
                self.log("tr_loss", loss)#, prog_bar=True)
                self.log("tr_loss_Y", loss_Y)#, prog_bar=True)
                self.log("tr_reg_l1", reg_l1)
                return loss
            def validation_step(batch, batch_idx):
                Xr, Xl, Sr, Sl, T, Y = batch
                _, H, zp = self(Xr, Xl, Sr, Sl, T)
                loss, loss_Y, reg_l1 = self._loss_Y(H,Y)
                self.log("val_loss", loss)#, prog_bar=True)
                self.log("val_loss_Y", loss_Y)#, prog_bar=True)
                self.log("val_reg_l1", reg_l1)
            self.training_step = training_step
            self.validation_step = validation_step
            mytrainer2 = self._get_trainer()
            mytrainer2.fit(model=self, train_dataloaders=loader_tr, val_dataloaders=loader_va)
            
            self.training_step = old_training_step
            self.validation_step = old_validation_step
            self.on_validation_epoch_end = old_on_validation_epoch_end
            
        else:
            self.mytrainer = self._get_trainer()
            self.mytrainer.fit(model=self, train_dataloaders=loader_tr, val_dataloaders=loader_va)

        #TODO fit machine learning version of HMM(X) to Y
        
        # post-processing, get params in interpretable numpy format
        self.startprob_ = th2np(th.softmax(F.pad(self.state_prior_unnormalized, (1,0), "constant", 0), dim=0))
        self.transmat_ = th2np(th.softmax(F.pad(self.transition_matrix_unnormalized, (1,0,0,0), "constant", 0), dim=1))
        self.emissionprob_X_bernoulli_ = th2np(th.sigmoid(self.emission_X_bernoulli_logit))
        self.emission_gaussian_mean_ = th2np(self.emission_X_gaussian_mean)
        self.emission_gaussian_sd_ = th2np(th.exp(self.emission_X_gaussian_sd_log))
        self.emissionprob_S_ = th2np(th.softmax(F.pad(self.emission_S_unnormalized,(1,0,0,0), 'constant',0), dim=1))
        self.coef_ = th2np(self.coef_th_)
        self.intercept_ = th2np(self.intercept_th_)
            
        return self

    #TODO
    def postprocess_states(self, X, sleep_stages, good_state_thres=60):
        """
        re-arange states by it's association with AASM stages
        X: list
        sleep_stages: list
        """
        Zp = self.predict_proba_Z(X)

        # remove uncommon states
        n_state = Zp.shape[1]
        Z = np.argmax(np.concatenate(Zp,axis=0), axis=1)
        good_states = [i for i in range(n_state) if (Z==i).sum()>=good_state_thres]
        transmat = self.transmat_[good_state][:,good_states]
        transmat = transmat/transmat.sum(axis=1, keepdims=True)
        emission = self.emissionprob_[good_states]
        self.set_transmat(transmat)
        self.set_emission(emission)

        # re-arange
        Zp = self.predict_proba_Z(X)
        Z = np.argmax(np.concatenate(Zp,axis=0), axis=1)
        ss = np.concatenate(sleep_stages,axis=0)-1
        ids = pd.notna(Z)&pd.notna(ss)
        cf = confusion_matrix(Z[ids], ss[ids])
        cf = cf[:,:5]
        sort_val1 = np.argmax(cf,axis=1)
        sort_val2 = cf[range(len(cf)), sort_val1]
        new_states = np.lexsort((sort_val2, sort_val1))
        transmat = transmat[new_states][:,new_states]
        emission = emission[new_states]
        self.set_transmat(transmat)
        self.set_emission(emission)
        
    def decision_function(self, X, return_z=False):
        X = [(x-self.Xmean_)/self.Xscale_ for x in X]
        dataset = MyDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=0, shuffle=False, collate_fn=MyDataset.collate)
        outputs = self.mytrainer.predict(model=self, dataloaders=loader, ckpt_path='best')
        
        H = np.concatenate([x['H'] for x in outputs])
        if return_z:
            zp = sum([x['zp'] for x in outputs], [])
            return H, zp
        else:
            return H
        
    def predict_proba_Z(self, X):
        _, zp = self.decision_function(X, return_z=True)
        return zp
        
    def predict_Z(self, X):
        zp = self.predict_proba_Z(X)
        zp = [np.argmax(z, axis=1) for z in zp]
        return zp
        
    def predict_proba(self, X):
        H = self.decision_function(X)
        yp = sigmoid(H)
        return yp#np.c_[1-yp,yp]
        
    def predict(self, X):
        yp = self.predict_proba(X)#[:,1]
        yp = (yp>0.5).astype(int)
        return yp
    
    def score(self, X=None, y=None, return_components=False):
        #yp = self.predict_proba(X)[:,1]
        y = self.val_output_y
        yp = self.val_output_yp
        auc = []
        for i in range(self.Dy):
            yi = y[:,i]
            ypi = yp[:,i]
            ids = (~np.isnan(yi))&(~np.isnan(ypi))
            yi = yi[ids].astype(int)
            ypi = ypi[ids]
            auc.append( roc_auc_score(yi, ypi) )
        if return_components:
            return np.mean(auc), auc
        else:
            return np.mean(auc)
    
    #TODO
    def save(self, path, separate=False):
        if separate:
            th.save(self.state_dict(), path)
        else:
            shutil.copyfile(self.mytrainer.checkpoint_callbacks[0].best_model_path, path)
        
    #TODO
    #@classmethod
    #def load(cls, path):
    

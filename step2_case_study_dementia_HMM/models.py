import logging
import os
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from scipy.special import expit as sigmoid
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import lightning
lightning.fabric.utilities.seed.seed_everything(2023)


class MyDataset(Dataset):
    def __init__(self, X, y=None):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        else:
            return (self.X[idx], self.y[idx])
    
    @classmethod
    def collate(cls, batch, split_to_shorter=False):
        has_y = type(batch[0])==tuple
        if split_to_shorter:
            batch_x = []
            batch_y = []
            for v in batch:
                if has_y:
                    xx = np.array_split(v[0], 5, axis=0)
                    batch_x.extend(xx)
                    batch_y.extend([v[1]]*len(xx))
                else:
                    batch_x.extend(np.array_split(v, 5, axis=0))
        else:
            if has_y:
                batch_x = [v[0] for v in batch]
                batch_y = [v[1] for v in batch]
            else:
                batch_x = batch
        T = [len(x) for x in batch_x]
        maxT = max(T)
        x_pr = np.array([np.pad(x, ((0,maxT-len(x)), (0,0))) for x in batch_x])  # pad right
        x_pl = np.array([np.pad(x, ((maxT-len(x),0), (0,0))) for x in batch_x])  # pad left
        if has_y:
            return (th.tensor(x_pr).float(), th.tensor(x_pl).float(), th.tensor(T).long(), th.tensor(batch_y).long())
        else:
            return (th.tensor(x_pr).float(), th.tensor(x_pl).float(), th.tensor(T).long())
        

def th2np(x, dtype=float):
    assert type(x) in [th.Tensor, nn.Parameter]
    return x.data.cpu().numpy().astype(dtype)


def log_domain_matmul(log_A, log_B):
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


def maxmul(log_A, log_B):
	"""
	log_A : m x n
	log_B : n x p
	output : m x p matrix

	Similar to the log domain matrix multiplication,
	this computes out_{i,j} = max_k log_A_{i,k} + log_B_{k,j}
	"""
	m = log_A.shape[0]
	n = log_A.shape[1]
	p = log_B.shape[1]

	log_A_expanded = th.stack([log_A] * p, dim=2)
	log_B_expanded = th.stack([log_B] * m, dim=0)

	elementwise_sum = log_A_expanded + log_B_expanded
	out1, out2 = th.max(elementwise_sum, dim=1)

	return out1, out2


#TODO make it generalizable, OOSSClassifier --> HMMOOSSClassifier, TransformerOOSClassifier
class HMMOOSSClassifier(BaseEstimator, ClassifierMixin, LightningModule):
    def __init__(self, thres_bounds=None, n_features=None, Xnames=None, n_components=3, C_l1=0.01, C_Y=1, C_emission=0.01, lr=0.001, max_iter=100, batch_size=8, lr_reduce_patience=3, early_stop_patience=10, random_state=None, verbose=True, warm_start_model_folder=''):
        super().__init__()
        self.save_hyperparameters()
        self.thres_bounds = thres_bounds
        self.n_features = n_features
        self.Xnames = Xnames
        self.n_components = n_components
        self.C_l1 = C_l1
        self.C_Y = C_Y
        self.C_emission = C_emission
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr_reduce_patience = lr_reduce_patience
        self.early_stop_patience = early_stop_patience
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start_model_folder = warm_start_model_folder
        self.validation_step_outputs = []
        
        if verbose:
            logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.INFO)
            logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.INFO)
        else:
            logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
            logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
    
    def _forward_log_probs(self, X, T, log_state_priors, log_transition_matrix, log_emission_matrix):
        """
        forward
        alpha(Z_t) = P(X_t|Z_t) x sum_{Z_t-1} alpha(Z_t-1) P(Z_t | Z_t-1)
        """
        _, batch_size, Tmax, D = X.shape
        log_alpha = th.zeros(batch_size, Tmax, self.n_components, device=self.device)
        for t in range(0, Tmax):
            if t==0:
                val2 = log_state_priors
            else:
                val2 = log_domain_matmul(log_transition_matrix, log_alpha[:, t-1].t()).t()
            val1 = th.tensordot(log_emission_matrix, X[:,:,t],dims=([1,2],[0,2])).t()
            log_alpha[:, t, :] = val1 + val2

        # Select the sum for the final timestep (each x may have different length).
        log_sums = log_alpha.logsumexp(dim=2)
        log_probs = th.gather(log_sums, 1, T.view(-1,1)-1).view(-1)/T  # /T to bring down its scale
        
        log_alpha = [log_alpha[i,:T[i]] for i in range(batch_size)]
        return log_probs, log_alpha
        
    def _get_z_log_probs(self, X, T, log_alpha, log_transition_matrix, log_emission_matrix):
        """
        backward
        beta(Z_t) = sum_{Z_t+1} beta(Z_t+1) x prod_d P(Xd_t+1|Z_t+1) x P(Z_t+1 | Z_t)
        """
        _, batch_size, Tmax, D = X.shape
        log_beta = th.zeros(batch_size, Tmax, self.n_components, device=self.device)
        for t in range(Tmax-2,-1,-1):
            val = th.tensordot(log_emission_matrix, X[:,:,t+1],dims=([1,2],[0,2])).t() + log_beta[:,t+1,:]
            log_beta[:,t,:] = log_domain_matmul(val, log_transition_matrix)
       
        zp = [F.softmax(log_alpha[i]+log_beta[i,Tmax-T[i]:], dim=1) for i in range(len(T))]
        return zp
        
    """ # not differentiable
    def _viterbi(self, X, T, log_state_priors, log_transition_matrix, log_emission_matrix):
        _, batch_size, Tmax, D = X.shape
        log_delta = th.zeros(batch_size, Tmax, self.n_components, device=self.device)
        psi = th.zeros(batch_size, Tmax, self.n_components, device=self.device).long()

        log_delta[:, 0, :] = th.tensordot(log_emission_matrix, X[:,:,0],dims=([1,2],[0,2])).t() + log_state_priors
        for t in range(1, Tmax):
            max_val, argmax_val = maxmul(log_transition_matrix, log_delta[:, t-1, :].t())
            max_val, argmax_val = max_val.t(), argmax_val.t()
            log_delta[:, t, :] = th.tensordot(log_emission_matrix, X[:,:,t],dims=([1,2],[0,2])).t() + max_val
            psi[:, t, :] = argmax_val

        # Get the log probability of the best path
        #log_max = log_delta.max(dim=2)[0]
        #best_path_scores = th.gather(log_max, 1, T.view(-1,1) - 1)

        # This next part is a bit tricky to parallelize across the batch,
        # so we will do it separately for each example.
        zp = []
        for i in range(0, batch_size):
            zp_i = [ log_delta[i, T[i]-1].max(dim=0)[1] ]
            for t in range(T[i]-1, 0, -1):
                z_t = psi[i, t, zp_i[0]]
                zp_i.insert(0, z_t)
            zp.append(th.stack(zp_i))
        return zp 
    """
            
    def forward(self, Xpr, Xpl, T, do_backward=True):
        """
        """
        batch_size, Tmax, D = Xpr.shape
        self.thres_bounds = self.thres_bounds.to(self.device)
        
        thres = th.sigmoid(self.unnormalized_thres)*(self.thres_bounds[1]-self.thres_bounds[0])+self.thres_bounds[0]
        Xpr_01 = th.sigmoid(10*(Xpr-thres))
        Xpr_01 = th.stack([1-Xpr_01, Xpr_01])
        Xpl_01 = th.sigmoid(10*(Xpl-thres))
        Xpl_01 = th.stack([1-Xpl_01, Xpl_01])

        log_state_priors = F.log_softmax(F.pad(self.unnormalized_state_priors, (1,0), "constant", 0), dim=0)
        log_transition_matrix = F.log_softmax(F.pad(self.unnormalized_transition_matrix, (0,0,1,0), "constant", 0), dim=0)
        log_emission_matrix = th.stack([
            F.logsigmoid(-self.unnormalized_emission_matrix),
            F.logsigmoid(self.unnormalized_emission_matrix),]).transpose(0,1)

        log_p_X_hmm, log_alpha = self._forward_log_probs(Xpr_01, T, log_state_priors, log_transition_matrix, log_emission_matrix)
        res = [log_p_X_hmm]
            
        if do_backward:
            #zp = self._viterbi(X01, T, log_state_priors, log_transition_matrix, log_emission_matrix)
            #zp_hist = th.vstack([th.bincount(z, minlength=self.n_components)/len(z) for z in zp])
            zp = self._get_z_log_probs(Xpl_01, T, log_alpha, log_transition_matrix, log_emission_matrix)
            zp_hist = th.vstack([z.mean(dim=0) for z in zp])
            zp_hist = zp_hist[:,:-1]
            H = th.matmul(zp_hist, self.coef_th_)+self.intercept_th_
            res.append(H)
            res.append(zp)
        if len(res)==1:
            res = res[0]
        return res
        
    def _loss_hmm(self, log_p_X_hmm):
        return -log_p_X_hmm.mean()
        
    def _loss_Y(self, H, Y):
        loss_Y = (-F.logsigmoid(H[Y==1]).sum()-F.logsigmoid(-H[Y==0]).sum())/len(Y)
        reg_l1 = th.mean(th.abs(self.coef_th_))
        return loss_Y + reg_l1*self.C_l1, loss_Y, reg_l1*self.C_l1
    
    def _loss(self, H, Y, log_p_X_hmm):
        loss_hmm = self._loss_hmm(log_p_X_hmm) 
        _, loss_Y, reg_l1 = self._loss_Y(H,Y)
        reg_emission = -th.mean((th.sigmoid(self.unnormalized_emission_matrix)-0.5)**2)
        return loss_hmm + loss_Y*self.C_Y  + reg_l1*self.C_l1 + reg_emission*self.C_emission, loss_hmm, loss_Y*self.C_Y, reg_l1*self.C_l1, reg_emission*self.C_emission
        
    def training_step(self, batch, batch_idx):
        Xpr, Xpl, T, Y = batch
        log_p_X_hmm, H, zp = self(Xpr, Xpl, T)
        loss, loss_hmm, loss_Y, reg_l1, reg_emission = self._loss(H, Y, log_p_X_hmm)
        self.log("train_loss", loss)
        self.log("train_loss_hmm", loss_hmm)
        self.log("train_loss_Y", loss_Y)
        self.log("train_reg_l1", reg_l1)
        self.log("train_reg_emission", reg_emission)
        return loss
        
    def validation_step(self, batch, batch_idx):
        Xpr, Xpl, T, Y = batch
        log_p_X_hmm, H, zp = self(Xpr, Xpl, T)
        loss, loss_hmm, loss_Y, reg_l1, reg_emission = self._loss(H, Y, log_p_X_hmm)
        self.log("val_loss", loss)
        self.log("val_loss_hmm", loss_hmm)
        self.log("val_loss_Y", loss_Y)
        res = {'H':H, 'zp':zp, 'Y':Y}
        self.validation_step_outputs.append(res)
        return res
        
    def predict_step(self, batch, batch_idx):
        Xpr, Xpl, T = batch
        log_p_X_hmm, H, zp = self(Xpr, Xpl, T)
        H = H.cpu().numpy().astype(float)
        zp = [z.cpu().numpy().astype(float) for z in zp]
        return {'H':H, 'zp':zp}
        
    def configure_optimizers(self):
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
        H = np.concatenate([th2np(x['H']) for x in self.validation_step_outputs]).astype(float)
        Y = np.concatenate([th2np(x['Y']) for x in self.validation_step_outputs]).astype(float)
        self.val_output_yp = sigmoid(H)
        self.val_output_y = Y
    
    def _get_trainer(self):
        return Trainer(
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
                
    def fit(self, X, Y, separate=False):
        path = os.path.join(self.warm_start_model_folder, f'n_state={self.n_components}.pth')
        if os.path.exists(path):
            weights = th.load(path)
            self.unnormalized_thres = nn.Parameter(weights['unnormalized_thres'])
            self.unnormalized_transition_matrix = nn.Parameter(weights['unnormalized_transition_matrix'])
            self.unnormalized_emission_matrix = nn.Parameter(weights['unnormalized_emission_matrix'])
            self.unnormalized_state_priors = nn.Parameter(weights['unnormalized_state_priors'])
            self.coef_th_ = nn.Parameter(weights['coef_'])
            self.intercept_th_ = nn.Parameter(weights['intercept_'])
            print(f'warm start weights loaded from {path}!')
        else:
            NC = self.n_components
            D = self.n_features
            scale = 1/np.sqrt(D)
            self.unnormalized_thres = nn.Parameter(th.zeros(D))
            tm = th.eye(NC)*2+th.randn(NC,NC)*scale
            tm = tm-tm[0]
            self.unnormalized_transition_matrix = nn.Parameter(tm[1:])
            self.unnormalized_emission_matrix = nn.Parameter(th.randn(NC,D)*scale)
            self.unnormalized_state_priors = nn.Parameter(th.randn(NC-1)*scale)
            self.coef_th_ = nn.Parameter(th.randn(NC-1)*scale)
            self.intercept_th_ = nn.Parameter(th.tensor(0.))
            
        assert len(X)==len(Y)
        N = len(Y)
        vaids = np.sort(np.random.choice(N, N//5, replace=False))
        trids = np.setdiff1d(np.arange(N), vaids)
        
        # standardize for better soft thresholding
        X2 = np.concatenate(X, axis=0)
        self.Xmean_ = np.nanmean(X2, axis=0)
        self.Xscale_ = np.nanstd(X2, axis=0)
        X = [(x-self.Xmean_)/self.Xscale_ for x in X]
        self.thres_bounds = (self.thres_bounds-self.Xmean_)/self.Xscale_
        self.thres_bounds = th.tensor(self.thres_bounds).float()
        
        dataset = MyDataset(X, Y)
        dtr = Subset(dataset, trids)
        loader_tr = DataLoader(dtr, batch_size=self.batch_size, num_workers=0, shuffle=True, collate_fn=MyDataset.collate)
        dva = Subset(dataset, vaids)
        loader_va = DataLoader(dva, batch_size=self.batch_size, num_workers=0, shuffle=False, collate_fn=MyDataset.collate)
        
        if separate:
            old_training_step = self.training_step
            old_validation_step = self.validation_step
            
            #collate_fn2 = lambda batch: MyDataset.collate(batch, split_to_shorter=True)
            #loader_tr.collate_fn = collate_fn2
            #loader_va.collate_fn = collate_fn2
            self.unnormalized_thres.requires_grad = False
            self.coef_th_.requires_grad = False
            self.intercept_th_.requires_grad = False
            def training_step(batch, batch_idx):
                Xpr, Xpl, T, Y = batch
                log_p_X_hmm = self(Xpr, Xpl, T, do_backward=False)
                loss = self._loss_hmm(log_p_X_hmm)
                self.log("train_loss", loss)
                return loss
            def validation_step(batch, batch_idx):
                Xpr, Xpl, T, Y = batch
                log_p_X_hmm = self(Xpr, Xpl, T, do_backward=False)
                loss = self._loss_hmm(log_p_X_hmm)
                self.log("val_loss", loss)
            self.training_step = training_step
            self.validation_step = validation_step
            mytrainer1 = self._get_trainer()
            mytrainer1.fit(model=self, train_dataloaders=loader_tr, val_dataloaders=loader_va)
            
            #loader_tr.collate_fn = MyDataset.collate
            #loader_va.collate_fn = MyDataset.collate
            self.unnormalized_transition_matrix.requires_grad = False
            self.unnormalized_emission_matrix.requires_grad = False
            self.unnormalized_state_priors.requires_grad = False
            self.coef_th_.requires_grad = True
            self.intercept_th_.requires_grad = True
            def training_step(batch, batch_idx):
                Xpr, Xpl, T, Y = batch
                _, H, zp = self(Xpr, Xpl, T)
                loss, loss_Y, reg_l1 = self._loss_Y(H,Y)
                self.log("train_loss", loss)
                self.log("train_loss_Y", loss_Y)
                self.log("train_reg_l1", reg_l1)
                return loss
            def validation_step(batch, batch_idx):
                Xpr, Xpl, T, Y = batch
                _, H, zp = self(Xpr, Xpl, T)
                loss, loss_Y, reg_l1 = self._loss_Y(H,Y)
                self.log("val_loss", loss)
                self.log("val_loss_Y", loss_Y)
                self.log("val_reg_l1", reg_l1)
            self.training_step = training_step
            self.validation_step = validation_step
            mytrainer2 = self._get_trainer()
            mytrainer2.fit(model=self, train_dataloaders=loader_tr, val_dataloaders=loader_va)
            
            self.training_step = old_training_step
            self.validation_step = old_validation_step
            
        else:
            self.mytrainer = self._get_trainer()
            self.mytrainer.fit(model=self, train_dataloaders=loader_tr, val_dataloaders=loader_va)
        
        # post-processing, get params in interpretable numpy format
        if type(self.thres_bounds)==th.Tensor:
            thres_bounds = th2np(self.thres_bounds)
        else:
            thres_bounds = np.array(self.thres_bounds)
        thres_bounds = thres_bounds*self.Xscale_+self.Xmean_
        self.X_thres_ = sigmoid(th2np(self.unnormalized_thres))*(thres_bounds[1]-thres_bounds[0])+thres_bounds[0]
        
        self.startprob_ = th2np(th.softmax(F.pad(self.unnormalized_state_priors, (1,0), "constant", 0), dim=0))
        self.transmat_ = th2np(th.softmax(F.pad(self.unnormalized_transition_matrix, (0,0,1,0), "constant", 0), dim=0)).T
        self.emissionprob_ = th2np(th.sigmoid(self.unnormalized_emission_matrix))
        
        self.coef_ = th2np(self.coef_th_)
        self.intercept_ = th2np(self.intercept_th_)
            
        return self
        
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
        return np.c_[1-yp,yp]
        
    def predict(self, X):
        yp = self.predict_proba(X)[:,1]
        yp = (yp>0.5).astype(int)
        return yp
    
    def score(self, X, y):
        #yp = self.predict_proba(X)[:,1]
        y = self.val_output_y
        yp = self.val_output_yp
        auc = roc_auc_score(y,yp)
        #emission_prob = sigmoid(th2np(self.unnormalized_emission_matrix))
        #emission_prob_divergence = np.mean(np.abs(emission_prob-0.5))
        return auc#+emission_prob_divergence
    
    #TODO
    def save(self, path, separate=False):
        if separate:
            th.save(self.state_dict(), path)
        else:
            shutil.copyfile(self.mytrainer.checkpoint_callbacks[0].best_model_path, path)
        
    #TODO
    #@classmethod
    #def load(cls, path):
    

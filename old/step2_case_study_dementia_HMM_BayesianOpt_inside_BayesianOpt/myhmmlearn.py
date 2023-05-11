import logging
import shutil
import numpy as np
#from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
#th.backends.cudnn.benchmark = False 
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import lightning
lightning.fabric.utilities.seed.seed_everything(2023)


class MyDataset(Dataset):
    def __init__(self, X, T):
        super().__init__()
        ids = np.r_[0, np.cumsum(T)]
        self.X = [X[ids[i]:ids[i+1]] for i in range(len(ids)-1)]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
    
    @classmethod
    def collate(cls, batch, split_to_shorter=True):
        if split_to_shorter:
            batch2 = []
            for x in batch:
                batch2.extend(np.array_split(x, 10, axis=0))
        else:
            batch2 = batch
        T = [len(x) for x in batch2]
        maxT = max(T)
        x_ = np.array([np.pad(x, ((0,maxT-len(x)), (0,0))) for x in batch2])
        return (th.tensor(x_).type('torch.LongTensor'), th.tensor(T).type('torch.LongTensor'))
        
    
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

	# log_A_expanded = th.stack([log_A] * p, dim=2)
	# log_B_expanded = th.stack([log_B] * m, dim=0)
    # fix for PyTorch > 1.5 by egaznep on Github:
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
	

class BernoulliHMM(LightningModule):
    """
    """
    def __init__(self, n_components=2, n_features=1, batch_size=16, lr=0.001, lr_reduce_patience=3, early_stop_patient=10, algorithm='viterbi', n_iter=100, tol=0.01, verbose=True):
        super().__init__()
        self.save_hyperparameters()
        
        NC = self.hparams.n_components
        D = self.hparams.n_features
        self.unnormalized_transition_matrix = nn.Parameter(th.eye(NC)*1+th.randn(NC,NC)*0.01)
        self.unnormalized_emission_matrix = nn.Parameter(th.randn(NC,D)*0.01)
        self.unnormalized_state_priors = nn.Parameter(th.randn(NC))
        if verbose:
            logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.INFO)
            logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.INFO)
        else:
            logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
            logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
    
    def forward(self, X, T):
        """
        x : IntTensor of shape (batch size, Tmax)
        T : IntTensor of shape (batch size)

        Compute log p(x) for each example in the batch.
        T = length of each example
        """
        batch_size, Tmax, D = X.shape

        log_state_priors = F.log_softmax(self.unnormalized_state_priors, dim=0)
        log_transition_matrix = F.log_softmax(self.unnormalized_transition_matrix, dim=0)
        log_emission_matrix = th.stack([
            F.logsigmoid(-self.unnormalized_emission_matrix),
            F.logsigmoid(self.unnormalized_emission_matrix),]).transpose(0,1)

        log_alpha = th.zeros(batch_size, Tmax, self.hparams.n_components, device=self.device)
        for t in range(0, Tmax):
            if t==0:
                val2 = log_state_priors
            else:
                val2 = log_domain_matmul(log_transition_matrix, log_alpha[:, t-1, :].t()).t()
            val1 = sum([log_emission_matrix[:,:,d][:,X[:,t,d]] for d in range(D)]).t()
            log_alpha[:, t, :] = val1 + val2

        # Select the sum for the final timestep (each x may have different length).
        log_sums = log_alpha.logsumexp(dim=2)
        log_probs = th.gather(log_sums, 1, T.view(-1,1) - 1)
        return -log_probs.mean()
        
    def training_step(self, batch, batch_idx):
        loss = self(batch[0], batch[1])
        self.log("train_loss", loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss = self(batch[0], batch[1])
        self.log("val_loss", loss)
        
    def predict_step(self, batch, batch_idx):
        """
        viterbi
        x : IntTensor of shape (batch size, Tmax)
        T : IntTensor of shape (batch size)
        Find argmax_z log p(x|z) for each (x) in the batch.
        """
        X, T = batch
        batch_size, Tmax, D = X.shape
        
        log_state_priors = F.log_softmax(self.unnormalized_state_priors, dim=0)
        log_transition_matrix = F.log_softmax(self.unnormalized_transition_matrix, dim=0)
        log_emission_matrix = th.stack([
            F.logsigmoid(-self.unnormalized_emission_matrix),
            F.logsigmoid(self.unnormalized_emission_matrix),]).transpose(0,1)
        
        log_delta = th.zeros(batch_size, Tmax, self.hparams.n_components, device=self.device).float()
        psi = th.zeros(batch_size, Tmax, self.hparams.n_components, device=self.device).long()

        log_delta[:, 0, :] = sum([log_emission_matrix[:,:,d][:,X[:,0,d]] for d in range(D)]).t() + log_state_priors
        for t in range(1, Tmax):
            max_val, argmax_val = maxmul(log_transition_matrix, log_delta[:, t-1, :].t())
            max_val, argmax_val = max_val.transpose(0,1), argmax_val.transpose(0,1)
            log_delta[:, t, :] = sum([log_emission_matrix[:,:,d][:,X[:,t,d]] for d in range(D)]).t() + max_val
            psi[:, t, :] = argmax_val

        # Get the log probability of the best path
        log_max = log_delta.max(dim=2)[0]
        best_path_scores = th.gather(log_max, 1, T.view(-1,1) - 1)

        # This next part is a bit tricky to parallelize across the batch,
        # so we will do it separately for each example.
        zp = []
        for i in range(0, batch_size):
            zp_i = [ log_delta[i, T[i] - 1, :].max(dim=0)[1].item() ]
            for t in range(T[i] - 1, 0, -1):
                z_t = psi[i, t, zp_i[0]].item()
                zp_i.insert(0, z_t)
            zp.append(np.array(zp_i))

        return {'z':zp, 'best_path_scores':best_path_scores.flatten()}
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=self.hparams.lr_reduce_patience, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=1e-7, verbose=self.hparams.verbose)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_loss",
            },
        }
    
    def _get_trainer(self):
        return Trainer(
                accelerator='auto', max_epochs=self.hparams.n_iter,
                deterministic=True, log_every_n_steps=3,
                check_val_every_n_epoch=1,
                callbacks=[
                    ModelCheckpoint(save_top_k=1, monitor='val_loss', mode='min'),
                    EarlyStopping(monitor="val_loss", mode="min", patience=self.hparams.early_stop_patient, min_delta=self.hparams.tol),
                    LearningRateMonitor(logging_interval='epoch')],
                    enable_progress_bar=self.hparams.verbose,
                    enable_model_summary=self.hparams.verbose,
                )
            
    def fit(self, X, T):
        assert len(X)==sum(T)
        N = len(T)
        vaids = np.sort(np.random.choice(N, N//5, replace=False))
        trids = np.setdiff1d(np.arange(N), vaids)
        
        dataset = MyDataset(X, T)
        dtr = Subset(dataset, trids)
        loader_tr = DataLoader(dtr, batch_size=self.hparams.batch_size, num_workers=4, shuffle=True, collate_fn=MyDataset.collate)
        dva = Subset(dataset, vaids)
        loader_va = DataLoader(dva, batch_size=self.hparams.batch_size, num_workers=4, shuffle=False, collate_fn=MyDataset.collate)
        
        mytrainer = self._get_trainer()
        mytrainer.fit(model=self, train_dataloaders=loader_tr, val_dataloaders=loader_va)
        self.hparams.model_path = mytrainer.checkpoint_callbacks[0].best_model_path
        return self
        
    def predict(self, X, T):
        assert len(X)==sum(T)
        
        dataset = MyDataset(X, T)
        collate_fn = lambda batch: MyDataset.collate(batch, split_to_shorter=False)
        loader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=4, shuffle=False, collate_fn=collate_fn)
        mytrainer = self._get_trainer()
        outputs = mytrainer.predict(model=self, dataloaders=loader, ckpt_path=self.hparams.model_path)
        
        zp = sum([x['z'] for x in outputs], [])
        zp = np.concatenate(zp)
        #best_path_scores = np.concatenate([x['best_path_scores'].cpu().numpy().astype(float) for x in outputs])
        return zp
    
    def save(self, path):
        shutil.copyfile(self.hparams.model_path, path)
        

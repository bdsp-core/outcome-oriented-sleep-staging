import pickle
from torch import nn
import torch as th
from torch.utils.data import Dataset


class OOSSDatset(Dataset):
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
        self.sids = res['sids']
        self.X = res['X']; self.Xnames = res['Xnames']
        self.S = res['S']
        self.Y = res['Y']
        #self.L = res['L']; self.Lnames = res['Lnames']

    def __getitem__(self, idx):
        return {'sid':self.sids[idx],
                'X':self.X[idx],
                'S':self.S[idx],
                'Y':self.Y[idx],
                #'L':self.L[idx],
                }

    def __len__(self):
        return len(self.sids)


class OOSSNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = 
        self.decoder = 

    def forward(self, x):
        return self.encoder(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def myloss(self, x_hat, logscale, x):
        return

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(),
            'reconstruction': recon_loss.mean(),
            'kl': kl.mean(),
        })

        return elbo



## Derivation for REINFORCE VI

We first ignore the time subscript, i.e., denote $X_{1:T}$ as $X$. The ELBO $L$ is
$$
\begin{aligned}
&\log P(S,X)\ge L(p_\theta,q_\phi)\\
&= -D_{KL}\left[q(Z,Y|S,X)||p(Z,Y))\right ] + \mathbb{E}_{q(Z,Y|S,X)}\left[\log p(S,X|Z,Y)\right ] \\
&= \sum_{Z,Y}q(Z,Y|S,X)\log\frac{p(Z,Y)}{q(Z,Y|S,X)}+\sum_{Z,Y}q(Z,Y|S,X)\log p(S,X|Z,Y) \\
&= \sum_{Z,Y}q(Y|Z)q(Z|S,X)\left[\log p(Z|Y)+\log p(Y)-\log q(Y|Z)-\log q(Z|S,X)\right]+\sum_Zq(Z|S,X)\log p(S,X|Z)\\
&= \mathbb{E}_{q(Z|S,X)}\left[\sum_Yq(Y|Z)\left[\log p(Z|Y)+\log p(Y)-\log q(Y|Z)-\log q(Z|S,X)\right]+\log p(S,X|Z)\right] \\
&=\mathbb{E}_{q_\phi(Z|S,X)}\left[f_{\theta,\phi}(Z)\right] \;.
\end{aligned}
$$

Differentiating $L$ w.r.t. $\phi$ is
$$
\begin{aligned}
\nabla_\phi L &= \nabla_\phi\mathbb{E}_{q_\phi(Z|S,X)}\left[f_{\theta,\phi}(Z)\right]\\
&=\mathbb{E}_{q_\phi(Z|S,X)}\left[(\nabla_\phi\log q_\phi(Z|S,X))f_{\theta,\phi}(Z)+\nabla_\phi f_{\theta,\phi}(Z)\right]\\
&=\mathbb{E}_{q_\phi(Z|S,X)}\left[\nabla_\phi(\log q_\phi(Z|S,X)\overline{f_{\theta,\phi}(Z)}+f_{\theta,\phi}(Z))\right] \;.
\end{aligned}
$$

Differentiating $L$ w.r.t. $\theta$ is
$$
\begin{aligned}
\nabla_\theta L &= \nabla_\theta\mathbb{E}_{q_\phi(Z|S,X)}\left[f_{\theta,\phi}(Z)\right]\\
&= \mathbb{E}_{q_\phi(Z|S,X)}\left[\nabla_\theta f_{\theta,\phi}(Z)\right]\\
&=\mathbb{E}_{q_\phi(Z|S,X)}\left[\nabla_\theta(\log q_\phi(Z|S,X)\overline{f_{\theta,\phi}(Z)}+f_{\theta,\phi}(Z))\right] \;.
\end{aligned}
$$

Therefore, the surrogate loss function is
$$
\frac{1}{M}\sum_{z_m\sim q_\phi(Z|S,X)}\log q_\phi(Z|S,X)\overline{f_{\theta,\phi}(Z)}+f_{\theta,\phi}(Z) \;,
$$where
$$
\begin{aligned}
f_{\theta,\phi}(Z)&=\sum_Yq_\phi(Y|Z)\left[\log p_\theta(Z|Y)+\log p_\theta(Y)-\log q_\phi(Y|Z)-\log q_\phi(Z|S,X)\right]+\log p_\theta(S,X|Z) \\
&=\sum_Yq_\phi(Y|Z)\left[\log p_\theta(Z|Y)+\log p_\theta(Y)-\log q_\phi(Y|Z)\right]-\log q_\phi(Z|S,X)+\log p_\theta(S,X|Z) \;.
\end{aligned}
$$

We now add back the Markov temporal structure, the surrogate loss function is
$$
\frac{1}{M}\sum_{z_m\sim \prod_t q_\phi(Z_t|S_t,X_t,Z_{t-1})} \sum_t\log q_\phi(Z_t|S_t,X_t,Z_{t-1})\overline{f_{\theta,\phi}(Z_{1:T})}+f_{\theta,\phi}(Z_{1:T}) \;,
$$where
$$
f_{\theta,\phi}(Z_{1:T})=\sum_Yq_\phi(Y|Z_{1:T})\left[\sum_t\log p_\theta(Z_t|Y,Z_{t-1})+\log p_\theta(Y)-\log q_\phi(Y|Z_{1:T})\right]-\sum_t\log q_\phi(Z_t|S_t,X_t,Z_{t-1})+\sum_t\log p_\theta(S_t,X_t|Z_t) \;.
$$

However, this estimator tends to have high variance. There are two approaches that can be used in combination.

First, remove non-downstream of $Z_s$ when estimating $\overline{f_{\theta,\phi}(Z_{1:T})}$ in $\log q_\phi(Z_s|S_s,X_s,Z_{s-1})\overline{f_{s,\theta,\phi}(Z_{1:T})}$
$$
\overline{f_{s,\theta,\phi}(Z_{1:T})}=\sum_Y\cdots-\sum_{t=s+1:T}\log q_\phi(Z_t|S_t,X_t,Z_{t-1})+\sum_{t=s+1:T}\log p_\theta(S_t,X_t|Z_t) \;.
$$

Second, use baseline $b$, i.e., a running average of recent samples of $\overline{f_{s,\theta,\phi}(Z_{1:T})}$
$$
\log q_\phi(Z_s|S_s,X_s,Z_{s-1})\left(\overline{f_{s,\theta,\phi}(Z)}-b\right) \;.
$$

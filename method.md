## Derivation for REINFORCE VI

The ELBO $L$ is
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
f_{\theta,\phi}(Z)=\sum_Yq(Y|Z)\left[\log p(Z|Y)+\log p(Y)-\log q(Y|Z)-\log q(Z|S,X)\right]+\log p(S,X|Z) \;.
$$

However, this estimator tends to have high variance.

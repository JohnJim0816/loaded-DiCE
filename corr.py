import numpy as np
import torch
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import mdptoolbox.example
import pickle

def magic_box(x):
    """Magic Box Operator. x are log-probabilities
    :param x:
    """
    return torch.exp(x - x.detach())

def gae(rewards, values, mask, tau=1.0, gamma_weighted=False):
    """
    Generalised Advantage Estimation https://arxiv.org/abs/1506.02438
    """
    values = values * mask
    deltas = rewards + gamma * values[:,1:] - values[:,:-1]
    advantages = torch.zeros_like(deltas).float()
    gae = torch.zeros_like(deltas[:,0]).float()
    for i in range(deltas.size(1) - 1, -1, -1):
        gae = gae * gamma * tau + deltas[:, i]
        advantages[:,i] = gae
    if gamma_weighted:
        gamma_weights = torch.cumprod(torch.ones_like(advantages) * gamma, 1) / gamma
        advantages = advantages * gamma_weights
    return advantages

def make_objective(batch_data, v_fn, tau=1.0, dice_lambda=1.0, use_dice=False, gamma_weighted=True):
    """Make objective for DiCE, Loaded DiCE, or classic surrogate loss
    """
    (batch_states, batch_pi_taken, batch_a_taken,
        batch_r, batch_dones, batch_mask, ep_returns) = batch_data
    empty_mask = (1 - batch_mask[:,:-1]).type(torch.ByteTensor)
    batch_pi_taken[empty_mask] = 1.0
    log_pi = torch.log(batch_pi_taken)
    batch_values = v_fn(batch_states).detach()
    advantages = gae(batch_r, batch_values, batch_mask, tau=tau, gamma_weighted=gamma_weighted)
    if use_dice == "old":
        log_pi_cumsum = torch.cumsum(log_pi, 1)
        deps = magic_box(log_pi_cumsum)
        batch_r[:,-1] = batch_r[:,-1] + batch_values[:,-1]*gamma
        if gamma_weighted:
            gamma_weights = torch.cumprod(torch.ones_like(advantages) * gamma, 1) / gamma
            batch_r = batch_r * gamma_weights
        obj = (batch_r * deps).sum(1).mean()
    elif use_dice == "loaded":
        weighted_cumsum = torch.zeros_like(log_pi)
        weighted_cumsum[:,0] = log_pi[:,0]
        for t in range(1, log_pi.size(1)):
            weighted_cumsum[:,t] = dice_lambda * weighted_cumsum[:,t-1] + log_pi[:,t]
        deps_exclusive = weighted_cumsum - log_pi
        full_deps = magic_box(weighted_cumsum) - magic_box(deps_exclusive)
        obj = (advantages * full_deps).sum(1).mean()
    else:
        obj = (advantages * log_pi).sum(1).mean()
    return obj

def get_P_pi(P, pi):
    """State transition function induced by MDP transition P and policy pi"""
    return (pi.softmax(0).unsqueeze(-1) * P).sum(0)

def get_V(S0, P_pi, R, gamma):
    """Analytically compute V for small MDPs"""
    return (S0 @ torch.inverse(torch.eye(S0.size(0)) - gamma*P_pi) * R).sum()

def sample(batch_size, max_steps=50, params=None):
    """Sample a batch of episodes from the MDP
    """
    ep_returns = []
    if params is None:
        params = pi

    batch_states = torch.zeros([batch_size, max_steps+1])
    batch_pi_taken = torch.zeros([batch_size, max_steps])
    batch_a_taken = torch.zeros([batch_size, max_steps])
    batch_r = torch.zeros([batch_size, max_steps])
    batch_dones = torch.zeros([batch_size, max_steps])
    batch_mask = torch.ones([batch_size, max_steps+1])
    
    state = Categorical(S0.unsqueeze(0).repeat(batch_size,1)).sample()
    
    for t in range(max_steps):
        r = R[state]
        probs = params[:,state].softmax(0).transpose(0,1)
        actions = Categorical(probs).sample()
        pi_taken = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        p_next = P[actions,state]
        next_state = Categorical(p_next).sample()

        batch_states[:,t] = state
        batch_pi_taken[:,t] = pi_taken
        batch_r[:,t] = r
        state = next_state
    batch_states[:, -1] = state

    return batch_states, batch_pi_taken, batch_a_taken, batch_r, batch_dones, batch_mask, ep_returns

np.random.seed(0)
torch.manual_seed(0)
n_states = 5
n_actions = 4
S0 = torch.ones(n_states) / n_states
P, R = mdptoolbox.example.rand(n_states, n_actions)
P = torch.tensor(P, dtype=torch.float)
R = (torch.randn(n_states) + 5) * 10
noise = 0.0
gamma = 0.95
pi = torch.randn(n_actions, n_states, requires_grad=True)

P_pi = get_P_pi(P, pi)
V = get_V(S0, P_pi, R, gamma)
V_s = torch.zeros(n_states)
for s in range(n_states):
    V_s[s] = get_V(torch.eye(n_states)[s], P_pi, R, gamma).detach() + torch.randn_like(V) * noise
    
v_fn = lambda s: V_s[s.long()].detach()

# batch_sizes = [4**n for n in range(3,9)] # bigger batches take longer, show estimator unbiasedness
batch_sizes = [4**n for n in range(3,7)]
n_orders = 3
n_seeds = 5

def corr_obj(tau, dice_method, lam=1.0):
    """Get derivatives up to n_orders
    :param tau: float: trades off bias and variance, see GAE
    :param dice_method: string: indicated which method uses, "old" or "loaded"
    :param lam: float: discount factor
    :return data: dictionary:
    """
    data = {}  # bs: [order: [seed1, ...]]
    print(f"{dice_method}, tau={tau}, lambda={lam}")
    for batch_size in batch_sizes:
        print("bs", batch_size)
        data[batch_size] = [[] for _ in range(n_orders)]
        for seed in range(n_seeds):
            batch_data = sample(batch_size)
            obj = make_objective(batch_data, v_fn, tau, use_dice=dice_method,
                                 dice_lambda=lam, gamma_weighted=True)

            grad = torch.autograd.grad(obj, pi, retain_graph=True, create_graph=True)[0]
            data[batch_size][0].append(grad.detach().numpy().flatten())
            for order in range(1, n_orders):
                # We only differentiate one element of the derivative to the next order
                grad = torch.autograd.grad(grad[0][0], pi, retain_graph=True, create_graph=True)[0]
                data[batch_size][order].append(grad.detach().numpy().flatten())
    
    return data

def get_true_grads():
    true_grads = []
    true_grad = torch.autograd.grad(V, pi, retain_graph=True, create_graph=True)[0]
    true_grads.append(true_grad.detach().numpy().flatten())
    for order in range(n_orders - 1):
        true_grad = torch.autograd.grad(true_grad[0][0], pi, retain_graph=True, create_graph=True)[0]
        true_grads.append(true_grad.detach().numpy().flatten())
    return true_grads

def run_corr_exp():
    old_dice_data = corr_obj(1.0, "old") # DiCE
    betterbase_data = corr_obj(1.0, "loaded") # Mao et al. corresponds to Loaded DiCE with tau=1.0
    lambda_data = corr_obj(0.0, "loaded") # Loaded DiCE, tau=0.0
    lvc_data = corr_obj(1.0, "loaded", lam=0.0) # LVC corresponds to Loaded DiCE with tau=1.0, lamda=0.0
    true_grads = get_true_grads()
    full_data = [old_dice_data, betterbase_data, lambda_data, lvc_data, true_grads]

    # jj:write data 
    with open("./corr_resluts/corr_results_demo.pkl", "wb") as f:
        pickle.dump(full_data, f)

def plot_corr_exp():
    # jj:read data
    with open("./corr_resluts/corr_results_demo.pkl", "rb") as f:
        full_data = pickle.load(f)

    true_grads = full_data[-1]
    full_data = full_data[:-1]
    full_corr_data = []

    for data in full_data:
        corr_data = {}
        for bs in batch_sizes:
            corr_data[bs] = [{} for _ in range(n_orders)]
            for order in range(n_orders):
                corrs = []
                grads = []
                for seed in range(n_seeds):
                    corrs.append(np.corrcoef(data[bs][order][seed], true_grads[order])[0][1])
                    grads.append(data[bs][order][seed])
                grad_std = np.std(np.stack(grads), 0).mean()

                corr_mean = np.mean(corrs)
                corr_std = np.std(corrs)
                corr_sem = corr_std / np.sqrt(n_seeds)
                corr_data[bs][order]["mean"] = corr_mean
                corr_data[bs][order]["std"] = corr_std
                corr_data[bs][order]["sem"] = corr_sem
                corr_data[bs][order]["grad_std"] = grad_std
        full_corr_data.append(corr_data)

    fig, axes = plt.subplots(1, n_orders, figsize=(22,6))

    axes_labels = ["1st Order", "2nd Order", "3rd Order"]

    for order in range(n_orders):
        ax = axes[order]
        for i, method in enumerate(["DiCE", "Mao et al.", "Ours", "LVC"]):
            corr_data = full_corr_data[i]
            ys = [corr_data[bs][order]["mean"] for bs in batch_sizes]
            errs = [corr_data[bs][order]["sem"] for bs in batch_sizes]
            ax.errorbar(batch_sizes, ys, yerr=errs, linestyle="None", marker='x', label=method, markersize=12,
                        elinewidth=2, markeredgewidth=2)
        ax.set_xscale('log')
        ax.set_ylim(-0.2,1.1)
        ax.set_xlabel(axes_labels[order])
    axes[0].set_ylabel("Correlation with true derivative")
    axes[0].legend(loc = "lower right")

    fig.tight_layout()
    fig.savefig("./corr_resluts/corr_demo.png")

if __name__ == '__main__':
    # Check the correlation of different estimators with the true derivatives as a function of batch size
    run_corr_exp()
    plot_corr_exp()
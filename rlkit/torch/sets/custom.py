"""

"""
import torch


def min_pairwise_distance(
        X: torch.Tensor,
        Y: torch.Tensor,
        dim=0,
):
    difference = X[:, None, :] - Y[None, :, :]
    squared_difference = difference**2
    min_squared_difference, _ = torch.min(squared_difference, dim=dim)
    return min_squared_difference.mean()


def min_pairwise_negative_likelihood(
        p_x,
        X: torch.Tensor,
        dim=0,
):
    bs, zd = X.shape  # batch size and z_dim
    log_probs = p_x.distribution.base_dist.expand((bs, bs, zd)).log_prob(X[:, None, :])
    nll = - log_probs.sum(dim=-1)
    min_nll, _ = torch.min(nll, dim=dim)
    return min_nll.mean()

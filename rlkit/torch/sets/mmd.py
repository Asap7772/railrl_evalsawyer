"""

"""
import torch
import rlkit.torch.pytorch_util as ptu

def mmd_distance(
        X: torch.Tensor,
        Y: torch.Tensor,
        kernel='imq',
        p_z_stddev=None,
):
    if kernel == 'imq':
        return imq_kernel(X, Y, p_z_stddev)
    elif kernel == 'rbf':
        return rbf_kernel(X, Y, p_z_stddev)
    else:
        raise NotImplementedError(kernel)


def imq_kernel(
        X: torch.Tensor,
        Y: torch.Tensor,
        p_z_stddev,
):
    X = X / p_z_stddev
    Y = Y / p_z_stddev
    batch_size = X.size(0)
    h_dim = X.size(1)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x) + C / (C + dists_y)

        res1 = (1 - ptu.eye(batch_size).cuda()) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / batch_size
        stats = stats + res1 - res2

    return stats


def rbf_kernel(
        X: torch.Tensor,
        Y: torch.Tensor,
        p_z_stddev,
):
    X = X / p_z_stddev
    Y = Y / p_z_stddev
    batch_size = X.size(0)
    h_dim = X.size(1)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.01, .1, 1., 10., 100]:
        C = 2 * h_dim * 1.0 / scale
        res1 = torch.exp(-C * dists_x) + torch.exp(-C * dists_y)

        res1 = (1 - ptu.eye(batch_size).cuda()) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = torch.exp(-C * dists_c)
        res2 = res2.sum() * 2. / batch_size
        stats = stats + res1 - res2

    return stats
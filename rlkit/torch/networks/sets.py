"""
Set-based / permtuation invariant networks
"""

from rlkit.torch.networks import mlp


class AverageSetMlp(mlp.Mlp):
    """
    Average across dimension 0.
    """
    def forward(self, x):
        x = super().forward(x)
        return x.mean(dim=0, keepdim=True)


class AverageSetConcatMultiHeadedMlp(mlp.ConcatMultiHeadedMlp):
    """
    Average across dimension 0.
    """
    def forward(self, x, batch=False):
        outputs = super().forward(x)
        if batch:
            return outputs
        else:
            return tuple(x.mean(dim=0, keepdim=True) for x in outputs)

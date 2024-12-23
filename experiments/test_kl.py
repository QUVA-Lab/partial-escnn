from collections import defaultdict
from functools import partial
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
from escnn import gspaces
from escnn.group import *
from escnn.kernels.steerable_filters_basis import SteerableFiltersBasis
from torch.nn.functional import normalize
import sys

sys.path.append("..")
from escnn2.kernels.fouriernonliniearity import FourierPointwise, InvFourier
from escnn.group import o2_group, so2_group


def obtain_act(prob_type="uniform", fts=None):
    group = so2_group()
    L = 2
    L_out = 4
    in_irreps = group.bl_irreps(L)
    out_irreps = group.bl_irreps(L_out)
    act = gspaces.no_base_space(group)
    N = group.bl_regular_representation(2 * (L)).size

    activation = FourierPointwise(
        act,
        irreps=in_irreps,
        out_irreps=out_irreps,
        channels=1,
        function="softmax",
        type="regular",
        normalize=False,
        N=N,
    )

    if fts is None:
        if prob_type == "uniform":
            factor1, factor2 = 1, 0
        elif prob_type == "ones":
            factor1, factor2 = 1, 1
        fts = []
        in_size = 0
        for irrep in in_irreps:
            irrep = group.irrep(*irrep)
            d = irrep.size**2 // irrep.sum_of_squares_constituents
            if prob_type == "rand":
                ft = torch.rand(d) if irrep.is_trivial() else torch.rand(d)
            else:
                ft = (
                    torch.ones(d) * factor1
                    if irrep.is_trivial()
                    else torch.ones(d) * factor2
                )
            in_size += d
            fts.append(ft)
        fts_in = torch.cat(fts)
        print(fts_in)

    else:
        fts_in = torch.tensor(fts)

    fts_in = activation.in_type(normalize(fts_in.reshape(1, -1)))
    fts_out = activation(fts_in)
    return fts_in, fts_out, activation


def compute_kl_divergence(p, q):
    p_fts_in, p_fts_out, p_act = p
    q_fts_in, q_fts_out, q_act = q


if __name__ == "__main__":
    # group = so2_group()
    # L = 2
    # L_out = 4
    # in_irreps = group.bl_irreps(L)
    # out_irreps = group.bl_irreps(L_out)
    # act = gspaces.no_base_space(group)
    # N = group.bl_regular_representation(2 * (L)).size

    # activation = FourierPointwise(
    #     act,
    #     irreps=in_irreps,
    #     out_irreps=out_irreps,
    #     channels=1,
    #     function="softmax",
    #     type="regular",
    #     normalize=False,
    #     N=N,
    # )
    # fts = []
    # in_size = 0
    # for irrep in in_irreps:
    #     irrep = group.irrep(*irrep)
    #     d = irrep.size**2 // irrep.sum_of_squares_constituents
    #     ft = torch.randn(d) if irrep.is_trivial() else torch.randn(d)
    #     in_size += d
    #     fts.append(ft)

    # fts_in = torch.cat(fts)
    # print(fts_in)
    # out_size = 0
    # psi_inds = {}
    # for irrep in out_irreps:
    #     irrep = group.irrep(*irrep)
    #     d = irrep.size**2 // irrep.sum_of_squares_constituents
    #     psi_inds[irrep.id] = torch.tensor(range(out_size, out_size + d))
    #     out_size += d

    # fts_in = activation.in_type(normalize(fts_in.reshape(1, -1)))
    # fts_out = activation(fts_in)

    p = obtain_act("uniform")
    q = obtain_act("rand")
    compute_kl_divergence(p, q)

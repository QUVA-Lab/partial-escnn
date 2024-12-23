import sys
from escnn import nn, group

sys.path.append("..")
from escnn2.r2convolution import R2Conv
from escnn2.r3convolution import R3Conv
from escnn2.linear import Linear
from escnn2 import gspaces
from escnn2.kernels.irreps_mapping import IrrepsMapFourierBLact

import torch

act_o3 = gspaces.rot3dOnR3()
in_type_o3 = nn.FieldType(act_o3, [irrep for irrep in act_o3.irreps[:3]])
out_type_o3 = nn.FieldType(act_o3, [irrep for irrep in act_o3.irreps[:3]])

conv = R3Conv(
    in_type_o3,
    out_type_o3,
    kernel_size=3,
    bias=False,
    learnable_eq=True,
    L=2,
    L_out=4,
)

# conv.check_equivariance()


# conv.check_equivariance()

import os
import sys

# Get the absolute path of the directory the script is in
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to PATH
sys.path.append(parent_dir)

from escnn import nn, group
from escnn2.r2convolution import R2Conv
from escnn2.linear import Linear
from escnn2 import gspaces
from escnn2.kernels.irreps_mapping import IrrepsMapFourierBLact


# from escnn.nn.modules.pointconv.r2_point_convolution import R2PointConv
import torch


def get_kl_loss(irrepmaps):
    irrepmaps = {irrepmap.layer_id: irrepmap for irrepmap in irrepmaps}
    kl_loss_sum = 0
    for layer_id, irrepmap in irrepmaps.items():
        if layer_id == 0:
            kl_loss = irrepmap.kl_divergence(None)
        else:
            kl_loss = irrepmap.kl_divergence(irrepmaps[layer_id - 1])
        kl_loss_sum += kl_loss / len(irrepmaps)
    return kl_loss_sum


def get_irrepsmaps(model):
    irrepmaps = list(
        filter(lambda x: isinstance(x, IrrepsMapFourierBLact), model.modules())
    )
    return irrepmaps


if __name__ == "__main__":
    act_o2 = gspaces.rot2dOnR2(N=-1, maximum_frequency=10)
    in_type_o2 = nn.FieldType(act_o2, [act_o2.irreps[i] for i in range(4)])
    out_type_o2 = nn.FieldType(
        act_o2, [group.directsum([act_o2.irreps[i] for i in range(4)])]
    )
    act_so2 = gspaces.flipRot2dOnR2(N=-1)
    in_type_so2 = nn.FieldType(act_so2, [act_so2.irreps[0] for i in range(4)])
    out_type_so2 = nn.FieldType(
        act_so2, [group.directsum([act_so2.irreps[i] for i in range(4)])]
    )

    in_type_so22 = nn.FieldType(act_so2, [act_so2.irreps[i] for i in range(4)])
    out_type_so22 = nn.FieldType(
        act_so2, [group.directsum([act_so2.irreps[i] for i in range(6)])]
    )

    learnable_so2 = R2Conv(
        in_type_so2,
        out_type_so2,
        kernel_size=3,
        learnable_eq=True,
        bias=False,
        L=2,
        L_out=4,
    )

    learnable_so22 = R2Conv(
        in_type_so22,
        out_type_so22,
        kernel_size=3,
        learnable_eq=True,
        bias=False,
        L=2,
        L_out=4,
    )

    so2 = R2Conv(
        in_type_so2,
        out_type_so2,
        kernel_size=3,
        bias=False,
    )

    irrepmaps1 = get_irrepsmaps(learnable_so2)
    irrepmaps2 = get_irrepsmaps(learnable_so22)

    irrepmaps = irrepmaps1 + irrepmaps2

    get_kl_loss(irrepmaps)

    exit()

    # o2 = R2Conv(
    #     in_type_o2,
    #     out_type_o2,
    #     kernel_size=3,
    #     basisexpansion="blocks",
    #     bias=False,
    # )

    # learnable_o2 = R2Conv(
    #     in_type_o2,
    #     out_type_o2,
    #     kernel_size=3,
    #     basisexpansion="learn_eq_norm",
    #     bias=False,
    # )  # .to("cuda")

    # so2.weights = learnable_so2.weights

    # exit()

    so2.eval()
    learnable_so2.eval()

    # print(learnable_o2.weights)
    # print(o2.weights)

    # for name, param in learnablel_so2.named_parameters():
    #     print(name)
    #     print(param)
    input_t = in_type_so2(torch.randn(1, in_type_so2.size, 27, 27))  # .to("cuda")

    # print(f"regular conv filter")
    # with torch.no_grad():
    #     learnable_so2.weights[: so2.weights.shape[0]] = so2.weights
    #     out2 = so2(input_t).tensor
    #     out1 = learnable_so2(input_t).tensor
    #     # print(f"basissampler conv filter")
    #     out2 = so2(input_t).tensor

    # print(((out1 - out2) ** 2).mean())
    import numpy as np

    model_parameters = filter(lambda p: p.requires_grad, learnable_so2.parameters())
    # for p in model_parameters:
    #     print(p)
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    device = "cpu"
    optimizer = torch.optim.Adam(learnable_so2.parameters())
    learnable_so2 = learnable_so2.to(device)
    model_parameters = filter(lambda p: p.requires_grad, learnable_so2.parameters())
    # for p in model_parameters:
    #     print(p)
    outs = learnable_so2(input_t.to(device)).tensor
    loss = (outs).mean()
    loss.backward()
    outs = learnable_so2(input_t.to(device)).tensor

    from escnn2.kernels.irreps_mapping import IrrepsMapFourierBLact

    ### THIS WILL FAIL
    learnable_so2.check_equivariance()

    # so2.check_equivariance()

    ########################## BOTH OF THESE WILL SUCCEED
    # learnablel_o2.check_equivariance()

    # thing_point = R2PointConv(in_type, activation.out_type, width=2, n_rings=2)
    # thing_point.check_equivariance()

    ########################

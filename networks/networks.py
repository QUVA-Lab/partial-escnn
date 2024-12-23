import sys
from escnn import group
from escnn import gspaces
from escnn import nn

sys.path.append("..")
from escnn2.linear import Linear

import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


class O2SteerableMLP(nn.EquivariantModule):
    def __init__(
        self,
        out_features,
        learn_eq=False,
        normalise_basis=True,
        prelim=False,
        gated=True,
        one_eq=False,
        split=False,
        iteration=0,
        only_first=False,
    ):
        super(O2SteerableMLP, self).__init__()

        self.prelim = prelim
        self.learn_eq = learn_eq

        self._id_offset = 100 * iteration

        self._one_eq = one_eq

        if split:
            self._c_fac = [2, 2]
            eq_channels = 2
        else:
            self._c_fac = [2]
            eq_channels = 1

        self._eq_channels = eq_channels

        self.G = group.o2_group(maximum_frequency=12)

        self.act = gspaces.no_base_space(self.G)

        self.in_type = nn.FieldType(self.act, [self.G.standard_representation()])

        self.only_first = only_first

        id_generator = (
            lambda layer, channel: (channel * 10 + layer * (not self._one_eq))
            + self._id_offset
        )

        L = 4

        self.layers_eq = torch.nn.ModuleList()

        for channel in range(eq_channels):
            if not gated:
                activation_1 = nn.FourierELU(
                    self.act,
                    channels=self._c_fac[channel],
                    irreps=[(0, 0)] + [(1, l) for l in range(L + 1)],
                    type="regular",
                    N=64,
                )
            else:
                irreps = self._c_fac[channel] * [
                    group.directsum(
                        [self.act.trivial_repr for _ in range(L + 2)]
                        + [self.act.irreps[i] for i in range(L + 2)]
                    ),
                ]
                out_type = nn.FieldType(self.act, irreps)
                activation_1 = nn.GatedNonLinearityUniform(out_type)

            block_1 = nn.SequentialModule(
                Linear(
                    self.in_type,
                    activation_1.in_type,
                    learnable_eq=learn_eq,
                    prelim=prelim,
                    normalise_basis=normalise_basis,
                    layer_id=id_generator(0, channel),
                ),
                nn.IIDBatchNorm1d(activation_1.in_type),
                activation_1,
            )

            if self.only_first:
                learn_eq = False

            L = 3
            if not gated:
                activation_2 = nn.FourierELU(
                    self.act,
                    channels=self._c_fac[channel] * 2,
                    irreps=[(0, 0)] + [(1, l) for l in range(L + 1)],
                    type="regular",
                    N=64,
                )
            else:
                irreps = (self._c_fac[channel] * 2) * [
                    group.directsum(
                        [self.act.trivial_repr for _ in range(L + 1)]
                        + [self.act.irreps[i] for i in range(L + 1)]
                    ),
                ]
                out_type = nn.FieldType(self.act, irreps)
                activation_2 = nn.GatedNonLinearityUniform(out_type)

            block_2 = nn.SequentialModule(
                Linear(
                    block_1.out_type,
                    activation_2.in_type,
                    learnable_eq=learn_eq,
                    prelim=prelim,
                    normalise_basis=normalise_basis,
                    layer_id=id_generator(1, channel),
                ),
                nn.IIDBatchNorm1d(activation_2.in_type),
                activation_2,
            )

            self.out_type = nn.FieldType(
                self.act, [self.G.trivial_representation] * out_features
            )
            block_3 = Linear(
                block_2.out_type,
                self.out_type,
                learnable_eq=learn_eq,
                prelim=prelim,
                normalise_basis=normalise_basis,
                layer_id=id_generator(2, channel),
            )
            layers = [block_1, block_2, block_3]
            self.layers_eq.append(nn.SequentialModule(*layers))

    def forward(self, x: torch.Tensor):
        x = self.in_type(x)
        outs = [layer(x).tensor for layer in self.layers_eq]
        return sum(outs)

    def evaluate_output_shape(self, input_shape: tuple):
        shape = list(input_shape)
        assert len(shape) == 2, shape
        assert shape[1] == self.in_type.size, shape
        shape[1] = self.out_type.size
        return shape

    @property
    def network_name(self):
        name = self.__class__.__name__
        if self.learn_eq:
            name = "Learnable " + name
            if self.prelim:
                name = "Preliminary " + name
        return name


class MLP(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        fac = 1
        self.block_1 = torch.nn.Sequential(
            torch.nn.Linear(in_features, 2 * fac),
            torch.nn.BatchNorm1d(2 * fac),
        )
        self.block_2 = torch.nn.Sequential(
            torch.nn.Linear(2 * fac, 4 * fac),
            torch.nn.BatchNorm1d(4 * fac),
            torch.nn.ReLU(),
        )
        self.block_3 = torch.nn.Sequential(torch.nn.Linear(4 * fac, out_features))

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        return x

    @property
    def network_name(self):
        return "MLP"


def testEquivariance(eq_network, mlp_network):
    device = "cpu"
    eq_model = eq_network().to(device)
    input_features = eq_model.in_type.size
    output_features = eq_model.out_type.size
    b = 100
    x = eq_model.in_type(torch.randn(b, input_features))

    for network in [eq_network, mlp_network]:
        if isinstance(network, MLP):
            model = network(input_features, output_features).to(device)
        else:
            model = network(output_features).to(device)
            input_features = model.in_type.size
            output_features = model.out_type.size
            x = model.in_type(torch.randn(b, input_features))
        model.eval()
        input_size = model.in_type.size
        B = 100

        with torch.no_grad():
            y = model(x.to(device)).to("cpu")


if __name__ == "__main__":
    pass

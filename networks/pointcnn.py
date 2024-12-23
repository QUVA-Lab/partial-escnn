import os
import torch
import random
from glob import glob
from tqdm.auto import tqdm
import wandb
import sys

import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    MLP,
    fps,
    global_max_pool,
    radius,
)

import escnn
from escnn import gspaces
from escnn import group
from escnn.nn import FieldType

from torch_geometric.data import Batch
from torch_geometric.data import Data

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)


class SetAbstraction(torch.nn.Module):
    def __init__(self, ratio, r, in_type, params, config, first_conv=None):
        super().__init__()
        self.ratio = ratio
        self.r = r

        channel, freq, width, n_rings, N = params

        self.N = N
        self.first_conv = first_conv

        self.act = gspaces.rot3dOnR3()
        self.in_type = in_type

        self.activ = config.activ

        self.G: Group = self.act.fibergroup

        if config.activ == "Quotient":
            self.activation = QuotientFourierELU(
                self.act,
                (False, -1),
                channel,
                irreps=self.G.bl_sphere_representation(L=config.freq).irreps,
                grid=self.G.sphere_grid(type="thomson", N=N),
            )
        elif config.activ == "Norm":
            irreps = []
            for i in range(freq + 1):
                irreps += [self.act.irrep(i)] * channel
            norm_type = escnn.nn.FieldType(self.act, irreps)
            self.activation = escnn.nn.NormNonLinearity(norm_type)

        elif config.activ == "Gated":
            irreps = []
            for i in range(freq + 1):
                irreps += [self.act.irrep(i)] * channel
            gated_type = escnn.nn.FieldType(self.act, irreps)
            gates_type = FieldType(self.act, [self.act.trivial_repr] * len(gated_type))
            gate_in_type = gates_type + gated_type
            self.activation = escnn.nn.GatedNonLinearity1(gate_in_type)

        elif config.activ == "TensorProduct":
            irrep_tensor = [self.G.bl_sphere_representation(L=freq)] * channel
            tensor_type = escnn.nn.FieldType(self.act, irrep_tensor)

            self.activation = escnn.nn.TensorProductModule(tensor_type, tensor_type)

        if config.kernel == "normalized":
            if self.first_conv is not None:
                self.conv = NormalizedR3PointConv(
                    in_type, self.activation.in_type, width=width, n_rings=n_rings
                )
            else:
                self.coord_in_type = escnn.nn.FieldType(
                    self.act, [self.act.fibergroup.standard_representation()]
                )
                self.conv = NormalizedR3PointConv(
                    in_type + self.coord_in_type,
                    self.activation.in_type,
                    width=width,
                    n_rings=n_rings,
                )
        else:
            if self.first_conv is not None:
                self.conv = escnn.nn.R3PointConv(
                    in_type, self.activation.in_type, width=width, n_rings=n_rings
                )
            else:
                self.coord_in_type = escnn.nn.FieldType(
                    self.act, [self.act.fibergroup.standard_representation()]
                )
                self.conv = escnn.nn.R3PointConv(
                    in_type + self.coord_in_type,
                    self.activation.in_type,
                    width=width,
                    n_rings=n_rings,
                )

        self.norm_layer = escnn.nn.IIDBatchNorm3d(self.activation.in_type)
        self.out_type = self.activation.out_type

    def forward(self, data, inn):
        x = data.x.to(device)
        pos = data.coords.to(device)
        batch = data.batch.to(device)

        # set_seed(42)
        print(pos, batch)
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        ).to(device)

        edge_index = torch.stack([col, row], dim=0).to(device)

        pos, batch = pos[idx], batch[idx]

        if self.first_conv is not None:
            x = inn(x, data.coords).to(device)
            x = self.conv(x, edge_index, out_coords=pos.to(device))
            x = self.norm_layer(x)
        else:
            inn = inn + self.coord_in_type
            x = inn(torch.cat((x, data.coords), dim=1), data.coords)
            x = self.conv(x, edge_index, out_coords=pos.to(device))
            x = self.norm_layer(x)

        if self.activ == "Quotient":
            x, A, Ainv = self.activation(x)
            return Data(x=x.tensor, coords=pos, batch=batch).to(device), A, Ainv
        else:
            x = self.activation(x)
            return Data(x=x.tensor, coords=pos, batch=batch).to(device)

    def check_equivariance(self):
        O = group.octa_group()
        N = 256

        self.to(device)
        x = torch.randn(N, self.in_type.size).to(device)
        coords = torch.randn(N, 3).to(device)

        data = Namespace(
            x=x, coords=coords, batch=torch.zeros(x.shape[0], dtype=torch.int64)
        )

        rho = self.in_type.representation.restrict("octa")
        rho_out = self.activation.out_type.representation.restrict("octa")

        self.eval()

        errs = []
        with torch.no_grad():
            y = self(data, self.in_type)[0].x
            y = y.cpu().numpy()

            for g in O.elements:
                g_x = x @ torch.tensor(rho(g).T, dtype=x.dtype, device=device)
                g_coords = coords @ torch.tensor(
                    O.standard_representation(g).T, dtype=coords.dtype, device=device
                )

                g_data = Namespace(
                    x=g_x,
                    coords=g_coords,
                    batch=torch.zeros(x.shape[0], dtype=torch.int64),
                )

                y_g = self(g_data, self.in_type)[0].x
                y_g = y_g.cpu().numpy()

                g_y = y @ rho_out(g).T

                # print(g)
                err = np.fabs(g_y - y_g)
                # print(f'{np.mean(err):.3f} +- {np.std(err):.3f}')
                n = np.maximum(np.fabs(g_y), np.fabs(y_g))
                n[n < 1e-5] = 1.0
                rel_err = err / n
                print(f"{np.mean(rel_err):.7f} +- {np.std(rel_err):.7f}")

                errs.append(np.mean(rel_err))

        # plt.plot(errs)
        # plt.show()
        return errs


class LinearLayer(torch.nn.Module):
    def __init__(self, in_type, out_type, params, config):
        super(LinearLayer, self).__init__()

        channel, N = params

        self.G = group.so3_group()
        self.gspace = gspaces.no_base_space(self.G)
        in_type.gspace == out_type.gspace == gspaces.GSpace3D

        # the input contains the coordinates of a point in the 3d space
        self.in_type_0 = self.gspace.type(
            in_type.representation
        )  # self.gspace.type(self.G.standard_representation())
        self.out_type_0 = self.gspace.type(out_type.representation)

        self.in_type = in_type
        self.out_type = out_type
        self.activ = config.activ

        if config.activ == "Quotient":
            self.activation1 = QuotientFourierELU(
                self.gspace,
                (False, -1),
                channel,
                irreps=self.G.bl_sphere_representation(L=config.freq).irreps,
                grid=self.G.sphere_grid(type="thomson", N=N),
            )
            self.activation2 = QuotientFourierELU(
                self.gspace,
                (False, -1),
                channel,
                irreps=self.G.bl_sphere_representation(L=config.freq).irreps,
                grid=self.G.sphere_grid(type="thomson", N=N),
            )
            self.activation3 = QuotientFourierELU(
                self.gspace,
                (False, -1),
                channel,
                irreps=self.G.bl_sphere_representation(L=config.freq).irreps,
                grid=self.G.sphere_grid(type="thomson", N=N),
            )
        elif config.activ == "Norm":
            irreps = []
            for i in range(config.freq + 1):
                irreps += [self.gspace.irrep(i)] * channel
            norm_type = escnn.nn.FieldType(self.gspace, irreps)
            self.activation1 = escnn.nn.NormNonLinearity(norm_type)
            self.activation2 = escnn.nn.NormNonLinearity(norm_type)
            self.activation3 = escnn.nn.NormNonLinearity(norm_type)
        elif config.activ == "Gated":
            irreps = []
            for i in range(config.freq + 1):
                irreps += [self.gspace.irrep(i)] * channel
            gated_type = FieldType(self.gspace, irreps)
            gates_type = FieldType(
                self.gspace, [self.gspace.trivial_repr] * len(gated_type)
            )
            gate_in_type = gates_type + gated_type
            self.activation1 = escnn.nn.GatedNonLinearity1(gate_in_type)
            self.activation2 = escnn.nn.GatedNonLinearity1(gate_in_type)
            self.activation3 = escnn.nn.GatedNonLinearity1(gate_in_type)
        elif config.activ == "TensorProduct":
            irrep_tensor = [self.G.bl_sphere_representation(L=config.freq)] * channel
            tensor_type = FieldType(self.gspace, irrep_tensor)
            self.activation1 = escnn.nn.TensorProductModule(tensor_type, tensor_type)
            self.activation2 = escnn.nn.TensorProductModule(tensor_type, tensor_type)
            self.activation3 = escnn.nn.TensorProductModule(tensor_type, tensor_type)

        # BLOCK 1
        self.block1 = escnn.nn.SequentialModule(
            escnn.nn.Linear(self.in_type_0, self.activation1.in_type),
            escnn.nn.IIDBatchNorm1d(self.activation1.in_type),
        )
        # BLOCK 2
        self.block2 = escnn.nn.SequentialModule(
            escnn.nn.Linear(self.block1.out_type, self.activation2.in_type),
            escnn.nn.IIDBatchNorm1d(self.activation2.in_type),
        )
        # BLOCK 3
        self.block3 = escnn.nn.SequentialModule(
            escnn.nn.Linear(self.block2.out_type, self.activation3.in_type),
            escnn.nn.IIDBatchNorm1d(self.activation3.in_type),
        )

        self.block4 = escnn.nn.Linear(self.block3.out_type, self.out_type_0)

    def forward(self, data):
        x = data.x.to(device)
        pos = data.coords.to(device)
        batch = data.batch.to(device)

        x = self.in_type_0(x).to(device)
        ####### block 1
        x = self.block1(x)
        if self.activ == "Quotient":
            x, A1, Ainv1 = self.activation1(x)
        else:
            x = self.activation1(x)

        ####### block 2
        x = self.block2(x)
        if self.activ == "Quotient":
            x, A2, Ainv2 = self.activation2(x)
        else:
            x = self.activation2(x)

        ####### block 3
        x = self.block3(x)
        if self.activ == "Quotient":
            x, A3, Ainv3 = self.activation3(x)
        else:
            x = self.activation3(x)

        x = self.block4(x)
        x = self.out_type(x.tensor, pos)

        data = make_batch_(x, pos, batch).to(device)

        if self.activ == "Quotient":
            return (
                Data(x=x.tensor, coords=pos, batch=batch),
                self.out_type,
                [A1, A2, A3],
                [Ainv1, Ainv2, Ainv3],
            )
        else:
            return Data(x=x.tensor, coords=pos, batch=batch), self.out_type

    def check_equivariance(self):
        O = group.octa_group()
        N = 256

        self.to(device)
        x = torch.randn(N, self.in_type.size).to(device)
        coords = torch.randn(N, 3).to(device)

        data = Namespace(
            x=x, coords=coords, batch=torch.zeros(x.shape[0], dtype=torch.int64)
        )

        rho = self.in_type_0.representation.restrict("octa")
        rho_out = self.out_type.representation.restrict("octa")

        self.eval()

        errs = []
        with torch.no_grad():
            y = self(data)[0].x
            y = y.cpu().numpy()

            for g in O.elements:
                g_x = x @ torch.tensor(rho(g).T, dtype=x.dtype, device=device)
                g_coords = coords @ torch.tensor(
                    O.standard_representation(g).T, dtype=coords.dtype, device=device
                )

                g_data = Namespace(
                    x=g_x,
                    coords=g_coords,
                    batch=torch.zeros(x.shape[0], dtype=torch.int64),
                )

                y_g = self(g_data)[0].x
                y_g = y_g.cpu().numpy()

                g_y = y @ rho_out(g).T

                # print(g)
                err = np.fabs(g_y - y_g)
                # print(f'{np.mean(err):.3f} +- {np.std(err):.3f}')
                n = np.maximum(np.fabs(g_y), np.fabs(y_g))
                n[n < 1e-5] = 1.0
                rel_err = err / n
                print(f"{np.mean(rel_err):.7f} +- {np.std(rel_err):.7f}")

                errs.append(np.mean(rel_err))

        # plt.plot(errs)
        # plt.show()
        return errs


class GlobalEquivariantSetupAbstraction(torch.nn.Module):
    def __init__(self, in_type, params, config):
        super().__init__()

        self.act = gspaces.rot3dOnR3()
        self.G = group.so3_group()

        channel, freq, width, n_rings, N = params

        self.N = N

        G: Group = self.act.fibergroup

        self.in_type = in_type
        self.activ = config.activ

        if config.activ == "Quotient":
            self.activation = QuotientFourierELU(
                self.act,
                (False, -1),
                channel,
                irreps=self.G.bl_sphere_representation(L=freq).irreps,
                out_irreps=[(0,)],
                grid=self.act.fibergroup.sphere_grid(type="thomson", N=N),
            )
        elif config.activ == "Norm":
            irreps = []
            for i in range(1):
                irreps += [self.act.irrep(i)] * channel
            norm_type = FieldType(self.act, irreps)
            self.activation = escnn.nn.NormNonLinearity(norm_type)
            self.normPool = escnn.nn.NormPool(self.activation.out_type)
        elif config.activ == "Gated":
            irreps = []
            for i in range(1):
                irreps += [self.act.irrep(i)] * channel
            gated_type = FieldType(self.act, irreps)
            gates_type = FieldType(self.act, [self.act.trivial_repr] * len(gated_type))
            gate_in_type = gates_type + gated_type
            self.activation = escnn.nn.GatedNonLinearity1(gate_in_type)
            self.normPool = escnn.nn.NormPool(self.activation.out_type)
        elif config.activ == "TensorProduct":
            irrep_tensor = [self.G.bl_sphere_representation(L=freq)] * channel
            tensor_type = FieldType(self.act, irrep_tensor)
            self.activation = escnn.nn.TensorProductModule(tensor_type, tensor_type)
            self.normPool = escnn.nn.NormPool(self.activation.out_type)

        if config.kernel == "normalized":
            self.conv = NormalizedR3PointConv(
                self.in_type, self.activation.in_type, width=width, n_rings=n_rings
            )
        else:
            self.conv = escnn.nn.R3PointConv(
                self.in_type, self.activation.in_type, width=width, n_rings=n_rings
            )

        self.block = escnn.nn.SequentialModule(
            escnn.nn.IIDBatchNorm3d(self.activation.in_type)
        )

    def forward(self, data, inn, A=None):
        row = data.batch.to(device)
        col = torch.arange(len(data.batch)).to(device)

        edge_index = torch.stack([col, row], dim=0).to(device)
        x = data.x.to(device)
        pos = data.coords.to(device)
        pos = torch_geometric.nn.global_mean_pool(pos, data.batch.to(device)).to(device)

        x = inn(x, data.coords).to(device)

        x = self.conv(x, edge_index, out_coords=pos)
        x = self.block(x)  # this is just the batchnorm layer

        batch = torch.arange(x.shape[0])

        if self.activ == "Quotient":
            x, A, Ainv = self.activation(x)
            return x, batch, A, Ainv
        else:
            x = self.activation(x)
            x = self.normPool(x)
            return x, batch

    def check_equivariance(self):
        from argparse import Namespace

        print(self.activ)
        # print(self.activation.out_type)

        O = group.octa_group()
        N = 256

        self.to(device)
        x = torch.randn(N, self.in_type.size).to(device)
        coords = torch.randn(N, 3).to(device)

        data = Namespace(
            x=x, coords=coords, batch=torch.zeros(x.shape[0], dtype=torch.int64)
        )

        rho = self.in_type.representation.restrict("octa")
        # TODO!!! might be wrong if self.normPool is used
        rho_out = self.out_type.representation.restrict("octa")
        print(rho_out.irreps)

        self.eval()

        errs = []
        with torch.no_grad():
            y = self(data, self.in_type)[0].tensor
            y = y.cpu().numpy()

            for g in O.elements:
                g_x = x @ torch.tensor(rho(g).T, dtype=x.dtype, device=device)
                g_coords = coords @ torch.tensor(
                    O.standard_representation(g).T, dtype=coords.dtype, device=device
                )

                g_data = Namespace(
                    x=g_x,
                    coords=g_coords,
                    batch=torch.zeros(x.shape[0], dtype=torch.int64),
                )

                y_g = self(g_data, self.in_type)[0].tensor
                y_g = y_g.cpu().numpy()

                # just making sure the output is invariant
                g_y = y  # @ rho_out(g).T

                # print(g)
                err = np.fabs(g_y - y_g)
                # print(f'{np.mean(err):.3f} +- {np.std(err):.3f}')
                n = np.maximum(np.fabs(g_y), np.fabs(y_g))
                n[n < 1e-5] = 1.0
                rel_err = err / n
                print(f"{np.mean(rel_err):.7f} +- {np.std(rel_err):.7f}")

                errs.append(np.mean(rel_err))

        # plt.plot(errs)
        # plt.show()
        return errs


class Model2(torch.nn.Module):
    def __init__(
        self,
        set_abstraction_ratio_1,
        set_abstraction_radius_1,
        set_abstraction_ratio_2,
        set_abstraction_radius_2,
        set_abstraction_ratio_3,
        set_abstraction_radius_3,
        config,
    ):
        super().__init__()

        self.shared_with_linear = config.shared_with_linear
        self.fully_shared_A = config.fully_shared_A
        self.activ = config.activ

        self.act = gspaces.rot3dOnR3()
        self.in_type = escnn.nn.FieldType(
            self.act,
            [self.act.trivial_repr]
            + 2 * [self.act.fibergroup.standard_representation()],
        )

        # params = [channel, freq, width, n_rings]
        params = [
            config.channels_conv[0],
            config.freq,
            config.width1,
            config.n_rings,
            config.N,
        ]
        self.sa1_module = SetAbstraction(
            set_abstraction_ratio_1,
            set_abstraction_radius_1,
            self.in_type,
            params,
            config,
            first_conv=True,
        )

        params = [config.channels_mlp[0], config.N]
        self.linear1 = LinearLayer(
            self.sa1_module.out_type, self.sa1_module.out_type, params, config
        )

        params = [
            config.channels_conv[1],
            config.freq,
            config.width2,
            config.n_rings,
            config.N,
        ]
        self.sa2_module = SetAbstraction(
            set_abstraction_ratio_2,
            set_abstraction_radius_2,
            self.linear1.out_type,
            params,
            config,
        )

        params = [config.channels_mlp[1], config.N]
        self.linear2 = LinearLayer(
            self.sa2_module.out_type, self.sa2_module.out_type, params, config
        )

        params = [
            config.channels_conv[2],
            config.freq,
            config.width3,
            config.n_rings,
            config.N,
        ]
        self.sa3_module = SetAbstraction(
            set_abstraction_ratio_3,
            set_abstraction_radius_3,
            self.linear2.out_type,
            params,
            config,
        )

        params = [config.channels_mlp[2], config.N]
        self.linear3 = LinearLayer(
            self.sa3_module.out_type, self.sa3_module.out_type, params, config
        )

        params = [16, config.freq, config.width4, config.n_rings, config.N]
        self.sa4_module = GlobalEquivariantSetupAbstraction(
            self.linear3.out_type, params, config
        )

        self.mlp = MLP(
            [16, 288, 576, 10], dropout=config.dropout, norm="batch_norm", act="relu"
        )

    def forward(self, data):
        sa0_out = (data, self.in_type)

        if self.activ == "Quotient":
            data, A1, A1inv = self.sa1_module(*sa0_out)
            data, linear_out, A1_, A1inv1_ = self.linear1(data)

            data, A2, A2inv = self.sa2_module(data, linear_out)
            data, linear_out, A2_, A2inv_ = self.linear2(data)

            data, A3, A3inv = self.sa3_module(data, linear_out)
            data, linear_out, A3_, A3inv_ = self.linear3(data)

            sa4_out = self.sa4_module(
                data, linear_out
            )  # Global pooling layer, so no A's (for now)
            x, batch, A_pool, Ainv_pool = sa4_out

            return (
                self.mlp(x.tensor).log_softmax(dim=-1),
                [A1, A2, A3],
                [A1inv, A2inv, A3inv],
            )
        else:
            print(sa0_out)
            data = self.sa1_module(*sa0_out)
            data, linear_out = self.linear1(data)

            data = self.sa2_module(data, linear_out)
            data, linear_out = self.linear2(data)

            data = self.sa3_module(data, linear_out)
            data, linear_out = self.linear3(data)

            sa4_out = self.sa4_module(
                data, linear_out
            )  # Global pooling layer, so no A's (for now)
            x, batch = sa4_out

            return self.mlp(x.tensor).log_softmax(dim=-1)


def make_batch_(x, pos, batch):
    return Data(x=x.tensor, coords=pos, batch=batch)


def add_ones(points):
    B, N, D = points.shape

    torch_ones = torch.ones((B, N, 1))
    points = torch.concat((torch_ones, points), axis=2)

    return points


def make_batch(points, labels):
    data_list = []
    for l, point in enumerate(points):
        coords = point[:, 1:4]

        data = Data(x=point, y=labels[l], coords=coords)
        data_list.append(data)

    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("PointNet++")

    parser.add_argument("--batch_size", type=int, default=12)

    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")

    parser.add_argument(
        "--set_abstraction_ratio_1", type=float, default=0.17467136373842823
    )
    parser.add_argument(
        "--set_abstraction_radius_1", type=float, default=0.2894570019003251
    )

    parser.add_argument(
        "--set_abstraction_ratio_2", type=float, default=0.5348073048273999
    )
    parser.add_argument(
        "--set_abstraction_radius_2", type=float, default=0.27907824089634037
    )

    parser.add_argument(
        "--set_abstraction_ratio_3", type=float, default=0.5881821493651226
    )
    parser.add_argument(
        "--set_abstraction_radius_3", type=float, default=0.173739121735466
    )

    # parser.add_argument('--learning_rate', type = float, default = 1e-3, help = "learning rate")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.00000652482801829556,
        help="learning rate",
    )
    # parser.add_argument('--weight_decay', type = float, default = 0.001)
    parser.add_argument("--weight_decay", type=float, default=0.06427599392214325)

    parser.add_argument("--width1", type=float, default=0.36104029841458096)
    parser.add_argument("--width2", type=float, default=0.3200028248086736)
    parser.add_argument("--width3", type=float, default=0.3641535834318061)
    parser.add_argument("--width4", type=float, default=3)

    # parser.add_argument('--n_rings', type = int, default = 3)
    parser.add_argument("--n_rings", type=int, default=5)
    parser.add_argument("--freq", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument(
        "--N",
        type=int,
        default=32,
        help="Number of samples in the QuotientFourierELU grid",
    )

    # SetAbstractionLayers
    # channels
    parser.add_argument(
        "--channels_conv",
        nargs="+",
        type=int,
        help="number of channels in convolutional layers (in order)",
        # metavar='N',
        default=[16, 32, 64],
        required=False,
    )  # , dest='my_list', choices=range(1, 5))
    # LinearLayers
    # channels
    parser.add_argument(
        "--channels_mlp",
        nargs="+",
        type=int,
        help="number of channels in convolutional layers (in order)",
        # metavar='N',
        default=[32, 64, 128],
        required=False,
    )

    # EquivariantMLP
    # channel, frequency, activation type
    # activation type of the EquivariantMLP that generates A
    parser.add_argument(
        "--activ_type",
        type=str,
        default="norm",
        help="Nonlinearity type used in EquivariantMLP that generates A matrix",
    )
    parser.add_argument(
        "--c",
        type=int,
        default=8,
        help="Number of channels for each frequency in the nonlinear layer of EquivariantMLP (for gated or normNonlinear layers)",
    )
    parser.add_argument(
        "--freq_mlp",
        type=int,
        default=3,
        help="Frequency used in equivariantMLP nonlinear layers",
    )

    # Rotations for training and testing and to compute equivariance error
    parser.add_argument(
        "--rot_training", type=str, default=None, help="Rotation on training set"
    )
    parser.add_argument(
        "--rot_test", type=str, default=None, help="Rotation on test set"
    )

    parser.add_argument(
        "--kernel",
        type=str,
        default=None,
        help="To state whether the kernel is normalized or not",
    )

    # name of the run on wandb
    parser.add_argument(
        "--name",
        type=str,
        default="base model",
        help="Stands for the name of the run on wandb",
    )

    # if set to true, the A is computed once in the convolutional layer and the same sampling matrix is used in the following linear layer
    # To set this true, simply call --shared_with_linear and nothing else :)
    parser.add_argument(
        "--shared_with_linear",
        action="store_true",
        help="Stated whether the sampling matrix (A), which is computed in the convolutional layer, will be shared with the following linear layer",
    )

    parser.add_argument(
        "--fully_shared_A",
        action="store_true",
        help="Indicates that the sampling matrix A is computed once in the first convolutional layer and shared with the all following layers in the network",
    )

    parser.add_argument(
        "---wandb_mode",
        type=str,
        default="online",
        help="Indicates whether the wandb is activated or not",
    )

    parser.add_argument(
        "--activ", type=str, default="Gated", help="Nonlinearity used in the network"
    )

    args = parser.parse_args()

    sys.path.append("..")

    from data import ModelNetDataLoader

    data = ModelNetDataLoader(
        "../data/modelnet/data/",
        split="train",
        uniform=False,
        normal_channel=True,
        npoint=1024,
    )
    train_loader = torch.utils.data.DataLoader(
        data, batch_size=8, shuffle=True, drop_last=True
    )
    data2 = ModelNetDataLoader(
        "../data/modelnet/data/",
        split="test",
        uniform=False,
        normal_channel=True,
        npoint=1024,
    )
    test_loader = torch.utils.data.DataLoader(
        data2, batch_size=8, shuffle=True, drop_last=True
    )
    # make the model, data, and optimization problem
    model = Model2(0.748, 0.4817, 0.25, 0.35, 0.2447, 0.2, args).to(device)

    # from main_final import *

    for batch in train_loader:
        points, labels = batch
        points = add_ones(points)
        batch = make_batch(points, labels)
        model(batch)

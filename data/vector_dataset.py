import numpy as np
import torch
from torch.utils.data import Dataset


class VectorDataset(Dataset):
    def __init__(self, nr_vecs=10000, noise=True):
        self.nr_vecs = nr_vecs
        self.noise = noise
        self.data, self.labels = self.create_data()

    def create_data(self):
        # data = torch.random.uniform(-10, 10, (nr_vecs, 2))
        data = torch.FloatTensor(self.nr_vecs, 2).uniform_(-1, 1)
        data_complex = torch.complex(data[:, 0], data[:, 1])
        labels = torch.empty((self.nr_vecs, 2), dtype=torch.float32)
        norm = torch.linalg.norm(data, axis=1)[:, None]
        print(max(norm))

        # vec = torch.FloatTensor([[100, 100]])
        # norm1 = torch.linalg.norm(vec, axis=1)
        # vec_norm = vec / norm1

        angles = torch.angle(data_complex)[:, None]

        labels = torch.cat((norm, angles), axis=1)
        # data = data + torch.normal(
        #     mean=torch.zeros_like(data),
        #     std=torch.ones_like(data) * 0.01,
        # )
        return (
            data,
            labels.float(),
        )
        # return data, labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :], self.labels[index, :]

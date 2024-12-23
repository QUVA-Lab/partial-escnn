from unicodedata import digit
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from escnn import group, gspaces, nn


class MNIST_angle(Dataset):
    def __init__(
        self,
        train=True,
        mnist_transform=transforms.Normalize((0.1307,), (0.3081,)),
        pre_transform=transforms.ConvertImageDtype(torch.float32),
        max_val=9,
        imgs_per_label=1000,
    ):
        self.max_val = max_val
        self.group = group.so2_group()
        self.act = gspaces.rot2dOnR2(N=-1)
        self.field = nn.FieldType(self.act, [self.act.trivial_repr])

        self.mnist = datasets.MNIST(
            root=f"data/MNIST/{'train' if train else 'test'}",
            download=True,
            train=train,
        )
        self.mnist_transform = mnist_transform
        self.pre_transform = pre_transform
        self.img_per_label = imgs_per_label
        self.train = train
        (
            self.imgs,
            self.mnist_labels,
            self.rot_elements,
            self.rot_labels,
        ) = self._create_data()

    def __len__(self):
        return len(self.mnist_labels)

    def _create_data(self):
        labels = self.mnist.targets
        inds = []
        for val in range(self.max_val + 1):
            val_inds = (labels == val).nonzero(as_tuple=True)[0]
            inds += list(np.random.choice(val_inds, size=self.img_per_label))
        rot_elements = [self.group.sample() for _ in inds]
        rot_labels = np.asarray(
            [element.value for element in rot_elements],
            dtype=np.float32,
        ) % (2 * np.pi)
        return self.mnist.data[inds], labels[inds], rot_elements, rot_labels

    def __getitem__(self, index):
        img = self.imgs[index]
        h, w = img.shape
        mnist_label = self.mnist_labels[index]
        rot_label = self.rot_labels[index]
        img = self.pre_transform(img)
        if self.train:
            transformation = self.rot_elements[index]
            img = (
                self.field(img[None, None, :, :])
                .transform(self.rot_elements[index])
                .tensor
            )
        else:
            transformation = self.group.sample()
            rot_label = torch.tensor(transformation.value, dtype=torch.float32)
            img = self.field(img[None, None, :, :]).transform(transformation).tensor
        # img = img.view(1, 1, 28, 28)
        img = self.mnist_transform(img)
        return img.view(1, h, w), mnist_label, rot_label


class MNIST_Double(Dataset):
    def __init__(
        self,
        train=True,
        digit_transform=transforms.ConvertImageDtype(torch.float),
        number_transform=None,
        max_val=9,
        square=True,
        images_per_class=100,
        normalize=True,
    ):
        self.num_transform = number_transform
        self.digit_transform = digit_transform
        self.normalize = normalize
        self.train = train
        self.max_val = max_val
        self.square = square
        self.imgs_per_class = images_per_class
        self.mnist = datasets.MNIST(
            root=f"data/MNIST/{'train' if train else 'test'}",
            download=True,
            train=train,
        )

        self.img1, self.img2, self.labels = self._create_data()
        if self.normalize:
            mean, std = self._std()
            self._normalize = transforms.Normalize((mean,), (std,))

    def _create_data(self):
        im_1_indices, im_2_indices, targets = [], [], []
        for num in range(100):
            tens = num // 10
            digits = num % 10
            tens_inds = list((self.mnist.targets == tens).nonzero(as_tuple=True)[0])
            digit_inds = list((self.mnist.targets == digits).nonzero(as_tuple=True)[0])
            im_1_indices += list(np.random.choice(tens_inds, size=self.imgs_per_class))
            im_2_indices += list(np.random.choice(digit_inds, size=self.imgs_per_class))
            targets += [num] * self.imgs_per_class

        return (
            self.mnist.data[im_1_indices],
            self.mnist.data[im_2_indices],
            torch.LongTensor(targets),
        )

    def __getitem__(self, index):
        img1, img2 = (
            self.img1[index],
            self.img2[index],
        )
        img1 = self.digit_transform(img1[None, :, :])
        img2 = self.digit_transform(img2[None, :, :])
        combined = self._combine_images(img1[0], img2[0])
        return combined, self.labels[index]

    def _std(self):
        if self.square:
            return (
                0.5 * 0.1307,
                np.sqrt(0.3081),
            )
        else:
            return 0.1307, 0.3081

    def _combine_images(self, img1, img2):
        h, w = img1.shape
        if self.square:
            h_start = int(0.5 * h)
            h_end = int(1.5 * h)
            combined = torch.zeros((1, h * 2, w * 2), dtype=img1.dtype)
            combined[:, h_start:h_end, :w] = img1
            combined[:, h_start:h_end, w:] = img2
        else:
            combined = torch.zeros((1, h, 2 * w - 1), dtype=img1.dtype)
            combined[:, :, :w] = img1
            combined[:, :, w - 1 :] = img2

        if self.num_transform is not None:
            combined = self.num_transform(combined)
        if self.normalize:
            combined = self._normalize(combined)
        return combined

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    pass

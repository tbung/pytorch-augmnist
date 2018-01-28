from PIL import Image
import os
from functools import partial
import numpy as np

import torch
import torch.utils.data as data

from torchvision.datasets import MNIST

def shear(image, value, axis):
    image = image.squeeze()
    base = image.min()
    coordinates = torch.nonzero(image!=base).t().float()

    fill = image[image!=base].clone()

    rangey,rangex = image.size()

    half = image.size(axis) // 2
    coordinates[1 - axis] = torch.ceil(coordinates[1 - axis] + (coordinates[axis]-half)*value)

    if coordinates.max() < rangex and coordinates.min() >= 0:
        result = torch.sparse.FloatTensor(coordinates.long(), fill, image.size()).to_dense()
        result[result == 0] = base
        return result
    else:
        # print('Overflow')
        return image

class AugMNIST(MNIST):
    def __init__(self, batch_size, transform=None, download=False, generate=False):
        self.train = True
        self.root = os.path.expanduser('data')
        self.augmented_file = f'augmented_{batch_size}.pt'
        self.transform = transform
        self.batch_size = batch_size

        if download:
            self.download()

        if generate:
            self.generate()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download MNIST' +
                               ' and generate=True to generate the augmentations')

        self.data = torch.load(
            os.path.join(self.root, self.processed_folder, self.augmented_file)
        )

    def __getitem__(self, index):
        img = self.data[index]

        # img = img.view(1, 28, 28)
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return self.data.size(0)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.augmented_file))

    def generate(self):
        if self._check_exists():
            return

        mnist_data, _ = torch.load(
            os.path.join(self.root, self.processed_folder, self.training_file)
        )

        mnist_data = mnist_data[:5000, :, :]

        T = np.linspace(-0.5, 0.5, self.batch_size)

        def generate_interpolation():
            for t in T:
                yield torch.stack(list(map(partial(shear, value=t, axis=1),
                                           mnist_data.float())), dim=0)

        aug_data = torch.stack(list(generate_interpolation()),
                               dim=0).transpose(0, 1).contiguous()
        aug_data = aug_data.view(-1, 28, 28)

        with open(os.path.join(self.root, self.processed_folder, self.augmented_file), 'wb') as f:
            torch.save(aug_data, f)

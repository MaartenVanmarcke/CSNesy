from nesy.parser import parse_program, parse_clause

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from itertools import product
from torch.utils.data import default_collate

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def custom_collate(batch):
    batch = tuple(zip(*batch))
    return default_collate(batch[0]), batch[1], default_collate(batch[2])


class AdditionTask(Dataset):

    def __init__(self, n=2, train=True, n_classes=10, nr_examples=None):
        assert n == 2, "Only n=2 is supported at the moment"
        self.train = train

        # We iterate over the MNIST dataset to apply the transform
        self.original_images = []
        self.original_targets = []
        for x,y in  MNIST('data/MNIST/', train=train, download=True, transform=transform):
            if y < n_classes:
                self.original_images.append(x)
                self.original_targets.append(y)
        self.original_images = torch.stack(self.original_images)
        self.original_targets = torch.tensor(self.original_targets)
        self.n_classes = n_classes
        self.num_digits = n
        program_string = "addition(X,Y,Z) :- digit(X,N1), digit(Y,N2), add(N1,N2,Z).\n"
        program_string += "\n".join(
            [f"add({x}, {y}, {x + y})." for x in range(self.n_classes) for y in range(self.n_classes)])
        program_string += "\n"
        program_string += "\n".join(
            [f"nn(digit, tensor(images, {x}), {y}) :: digit(tensor(images, {x}),{y})." for x, y in
             product(range(self.num_digits), range(self.n_classes))])
        self.program = parse_program(program_string)

        if nr_examples is not None:
            if nr_examples > self.nr_examples:
                raise ValueError('nr_examples exceeds to number of available examples in this dataset')
            else:
                self.nr_examples = nr_examples
        else:
            self.nr_examples = len(self.original_images) // self.num_digits

    def __getitem__(self, index):
        images = self.original_images[index * self.num_digits: (index + 1) * self.num_digits]
        targets = self.original_targets[index * self.num_digits: (index + 1) * self.num_digits]
        target = int(targets.sum())

        if self.train:
            # In MNIST Addition, training queries for a single pair of images check for a given sum (i.e. the target)
            # Therefore, we have a List[Term], each element of the list correspond to a single pair of images

            query = parse_program("addition(tensor(images, 0), tensor(images,1), {}).".format(target))[0].term
            tensor_sources = {"images": images}

            return tensor_sources, query, torch.tensor([1.0])
        else:
            # In MNIST Addition, testing queries for a single pair of images check for all possible sums.
            # In this way, we can compute the most probable sum.
            # Therefore, we have a List[List[Term]], each element of the outer list correspond to a single pair of
            # images. Each element of the inner list correspond to a possible sum.

            queries = [parse_program("addition(tensor(images, 0), tensor(images,1), {}).".format(z))[0].term
                       for z in range(self.n_classes * 2 - 1)]
            tensor_sources = {"images": images}

            return tensor_sources, queries, target

    def dataloader(self, batch_size=2, shuffle=None, num_workers=0):
        if shuffle is None:
            shuffle = self.train

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate,
                          num_workers=num_workers)

    def __len__(self):
        return self.nr_examples

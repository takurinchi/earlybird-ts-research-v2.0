import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def create_loaders(root="data", batch_size=64, seed=1, num_workers=0):
    datasets.MNIST.mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/"]
    normalize = transforms.Normalize((0.1307), (0.3081))
    train_transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor(),normalize])
    eval_transform = ([transforms.ToTensor(), normalize])
    training = datasets.MNIST(root, train=True, download=True, transform=train_transform)
    validation_base = datasets.MNIST(root, train=True, download=True, transform=eval_transform)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(training), generator=generator).tolist()
    n_validation = 6000
    validation_indices, train_indices = permutation[:n_validation], permutation[n_validation:]
    test = datasets.MNIST(root, train=False, download=True, transform=eval_transform)
    kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    return{
        "train": DataLoader(Subset(training, train_indices), shuffle=True, **kwargs),
        "validation": DataLoader(Subset(validation_base, validation_indices), shuffle=False, **kwargs),
        "test": DataLoader(test, shuffle=False, **kwargs),
    }
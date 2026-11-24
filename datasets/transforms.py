"""Image transformations and data augmentations for DEEPVISION."""

import torchvision.transforms as T

# Standard normalization statistics
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_mnist_transforms(augment: bool = False) -> T.Compose:
    """Get transformation pipeline for MNIST dataset.

    Args:
        augment: If True, apply training data augmentations (random rotation, small crop).
                 If False, apply standard deterministic normalization only.

    Returns:
        T.Compose: Composed torchvision transform.
    """
    if augment:
        return T.Compose([
            T.RandomRotation(degrees=10),
            T.RandomCrop(28, padding=2, padding_mode="edge"),
            T.ToTensor(),
            T.Normalize(mean=MNIST_MEAN, std=MNIST_STD),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MNIST_MEAN, std=MNIST_STD),
        ])


def get_cifar10_transforms(augment: bool = False) -> T.Compose:
    """Get transformation pipeline for CIFAR-10 dataset (prepared for future phase).

    Args:
        augment: If True, apply standard CIFAR-10 augmentations.
                 If False, apply deterministic normalization only.

    Returns:
        T.Compose: Composed torchvision transform.
    """
    if augment:
        return T.Compose([
            T.RandomCrop(32, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])

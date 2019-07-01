import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def load_train_data(datadir, batch_size):
    train_trainsforms = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(), ])
    train_data = datasets.ImageFolder(datadir, transform=train_trainsforms)
    num_train = len(train_data)
    train_idx = list(range(num_train))
    np.random.shuffle(train_idx)
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_loader
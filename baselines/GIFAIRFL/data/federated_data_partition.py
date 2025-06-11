import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.transforms as T
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# make sure these imports match your package structure
from folktables import ACSDataSource, ACSEmployment, ACSIncome, generate_categories, ACSPublicCoverage, ACSMobility

# reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class AddGaussianNoise(object):
    
    def __init__(self, mean=0, std=1):
     
        self.mean = mean
        self.std = std

    def __call__(self, img):
     
        # Convert the image to a PyTorch tensor
        img = transforms.ToTensor()(img)
     
        # Generate Gaussian noise with the same size as the image
        noise = torch.randn(img.size()) * self.std + self.mean
     
        # Add the noise to the image
        noisy_img = img + noise
     
        # Clip the pixel values to be in the range [0, 1]
        noisy_img = torch.clamp(noisy_img, 0, 1)
     
        # Convert the noisy image back to a PIL Image
        noisy_img = transforms.ToPILImage()(noisy_img)
     
        return noisy_img

class SubsetWithTransform(Dataset):
    
    """Subset that applies its own transform, ignoring any base_dataset.transform."""
    def __init__(self, base_dataset, indices, transform=None):
        
        self.base = base_dataset
        self.idx  = indices
        self.tf   = transform

    def __len__(self):
        
        return len(self.idx)

    def __getitem__(self, i):
        
        x, y = self.base[self.idx[i]]
        
        return (self.tf(x), y) if self.tf else (x, y)

def get_transforms(dataset_name: str, n_clients: int):
    
    """
    Train‐time transforms: for each dataset we define 5 pipelines
    (varying only in Gaussian-noise std), then cycle them across clients.
    """
    name = dataset_name.lower()
    # the five noise levels

    # ---------- FashionMNIST / FMNIST ----------
    if name == "fmnist":
        
        fmnist_stds = [0.4, 0.5, 0.7, 1.0, 1.5]

        pipelines = []
        
        for s in fmnist_stds:
            
            pipelines.append(
                T.Compose([
                    T.Grayscale(num_output_channels=1),
                    AddGaussianNoise(0., s),
                    T.ToTensor(),
                    T.Normalize([0.406], [0.225])
                ])
            )
        
        return [pipelines[i % 5] for i in range(n_clients)]

    # ---------- MNIST ----------
    elif name == "mnist":

        mnist_stds = [0.4, 0.5, 0.7, 1.0, 1.5]
        pipelines = []
        
        for s in mnist_stds:
            
            pipelines.append(
                T.Compose([
                    AddGaussianNoise(0., s),
                    T.ToTensor(),
                    T.Normalize((0.1307,), (0.3081,))
                ])
            )
        
        return [pipelines[i % 5] for i in range(n_clients)]

    # ---------- CIFAR10 ----------
    elif name == "cifar10":
        
        mean, std = (0.4914,0.4822,0.4465), (0.2675,0.2565,0.2761)
        pipelines = []
        stds = [0.4, 0.5, 0.7, 1.0, 1.5]
        
        for s in stds:
            
            stds = [0.4, 0.5, 0.7, 1.0, 1.5]

            pipelines.append(
                T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomCrop(32, padding=4),
                    AddGaussianNoise(0., s),
                    T.ToTensor(),
                    T.Normalize(mean, std)
                ])
            )
        
        return [pipelines[i % 5] for i in range(n_clients)]

    # ---------- CIFAR100 ----------
    elif name == "cifar100":
        mean, std = (0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761)

        stds = [0.4, 0.5, 0.7, 1.0, 1.5]
        pipelines = []

        for s in stds:
            pipelines.append(
                T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomCrop(32, padding=4),
                    AddGaussianNoise(0., s),
                    T.ToTensor(),
                    T.Normalize(mean, std)
                ])
            )
        return [pipelines[i % 5] for i in range(n_clients)]

    # ---------- FER2013 ----------
    elif name == "fer":

        fer_stds = [0.0, 0.09, 0.18, 0.27, 0.36]
        pipelines = []
        
        for s in fer_stds:
            
            pipelines.append(
                T.Compose([
                    T.RandomCrop(44),
                    T.RandomHorizontalFlip(),
                    AddGaussianNoise(0., s),
                    T.ToTensor()
                ])
            )
        
        return [pipelines[i % 5] for i in range(n_clients)]

    # ---------- UTKFace ----------
    elif name == "utk" or name == "utkfaces":

        utk_stds = [0.0, 0.1, 0.3, 0.5, 0.7]
        pipelines = []
        
        for s in utk_stds:
            
            pipelines.append(
                T.Compose([
                    T.Resize((32,32)),
                    AddGaussianNoise(0., s),
                    T.ToTensor(),
                    T.Normalize((0.49,), (0.23,))
                ])
            )
        
        return [pipelines[i % 5] for i in range(n_clients)]

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_test_transforms(dataset_name: str, n_clients: int):
    
    """
    Test‐time pipelines: exactly the same five noise‐level variants,
    cycled across clients (no extra augmentations).
    """
    name = dataset_name.lower()
    
    # ---------- FashionMNIST ----------
    if name == "fmnist":

        fmnist_stds = [0.4, 0.5, 0.7, 1.0, 1.5]
        pipelines = []
        
        for s in fmnist_stds:
            
            pipelines.append(
                T.Compose([
                    T.Grayscale(1),
                    AddGaussianNoise(0., s),
                    T.ToTensor(),
                    T.Normalize([0.406], [0.225])
                ])
            )
        
        return [pipelines[i % 5] for i in range(n_clients)]

    # ---------- MNIST ----------
    elif name == "mnist":

        mnist_stds = [0.4, 0.5, 0.7, 1.0, 1.5]
        pipelines = []

        for s in mnist_stds:
            
            pipelines.append(
                T.Compose([
                    AddGaussianNoise(0., s),
                    T.ToTensor(),
                    T.Normalize((0.1307,), (0.3081,))
                ])
            )
        
        return [pipelines[i % 5] for i in range(n_clients)]

    # ---------- CIFAR10 ----------
    elif name == "cifar10":
        
        mean, std = (0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)
        pipelines = []
        stds = [0.4, 0.5, 0.7, 1.0, 1.5]
        
        for s in stds:
            
            pipelines.append(
                T.Compose([
                    AddGaussianNoise(0., s),
                    T.ToTensor(),
                    T.Normalize(mean, std)
                ])
            )
        return [pipelines[i % 5] for i in range(n_clients)]

    # ---------- CIFAR100 ----------
    elif name == "cifar100":
        
        mean, std = (0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761)
        pipelines = []
        stds = [0.4, 0.5, 0.7, 1.0, 1.5]
        
        for s in stds:
            
            pipelines.append(
                T.Compose([
                    AddGaussianNoise(0., s),
                    T.ToTensor(),
                    T.Normalize(mean, std)
                ])
            )
        
        return [pipelines[i % 5] for i in range(n_clients)]

    # ---------- FER2013 ----------
    elif name == "fer":

        fer_stds = [0.0, 0.09, 0.18, 0.27, 0.36]
        pipelines = []
        
        for s in fer_stds:
            
            pipelines.append(
                T.Compose([
                    T.RandomCrop(44),
                    T.RandomHorizontalFlip(),
                    AddGaussianNoise(0., s),
                    T.ToTensor()
                ])
            )
        
        return [pipelines[i % 5] for i in range(n_clients)]

    # ---------- UTKFace ----------
    elif name == "utk" or name == "utkfaces":

        utk_stds = [0.0, 0.1, 0.3, 0.5, 0.7]
        pipelines = []

        for s in utk_stds:
            
            pipelines.append(
                T.Compose([
                    T.Resize((32,32)),
                    AddGaussianNoise(0., s),
                    T.ToTensor(),
                    T.Normalize((0.49,), (0.23,))
                ])
            )
        
        return [pipelines[i % 5] for i in range(n_clients)]

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def load_dataset(name, train_transform=None, test_transform=None):
    
    n = name.lower()
    
    if n=="cifar10":
        
        train = datasets.CIFAR10("./data", train=True, download=True, transform=None)
        test  = datasets.CIFAR10("./data", train=False, download=True, transform=None)

    elif n=="cifar100":
        
        train = datasets.CIFAR100("./data", train=True, download=True, transform=None)
        test  = datasets.CIFAR100("./data", train=False, download=True, transform=None)

    elif n=="mnist":
        
        train = datasets.MNIST("./data", train=True, download=True, transform=None)
        test  = datasets.MNIST("./data", train=False, download=True, transform=None)

    elif n=="fmnist":
        
        train = datasets.FashionMNIST("./data", train=True, download=True, transform=None)
        test  = datasets.FashionMNIST("./data", train=False, download=True, transform=None)

    elif n=="fer":
        
        train = datasets.ImageFolder("/home/khotso/Individual-Subgroup-Fairness/data/FER2013/train", transform=None)
        test  = datasets.ImageFolder("/home/khotso/Individual-Subgroup-Fairness/data/FER2013/test",  transform=None)

    elif n in ("utk","utkfaces"):
        
        #from data.federated_data_partition import UTKFaceDataset
        
        train = datasets.ImageFolder("/home/khotso/Individual-Subgroup-Fairness/data/UTKFace/train", transform=None)
        test  = datasets.ImageFolder("/home/khotso/Individual-Subgroup-Fairness/data/UTKFace/test",  transform=None)

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return train, test

def partition_iid(targets, num_clients, seed=seed):
    
    np.random.seed(seed)
    idx = np.arange(len(targets))
    np.random.shuffle(idx)
    base, rem = divmod(len(idx), num_clients)
    out, st = {}, 0
    
    for c in range(num_clients):
        
        sz = base + (1 if c<rem else 0)
        out[c] = idx[st:st+sz].tolist()
        st += sz
    
    return out

def partition_dirichlet(targets, num_clients, num_classes, alpha, seed=seed):
    
    np.random.seed(seed)
    targets = np.array(targets)
    out = {c:[] for c in range(num_clients)}
    
    for cls in range(num_classes):
        
        cls_idx = np.where(targets==cls)[0]
        np.random.shuffle(cls_idx)
        if len(cls_idx)==0: continue
        ps = np.random.dirichlet([alpha]*num_clients)
        cnt = np.random.multinomial(len(cls_idx), ps)
        st = 0
        for c,ct in enumerate(cnt):
            if ct>0:
                out[c].extend(cls_idx[st:st+ct].tolist())
                st += ct
    return out

def partition_data(targets, method, num_clients, alpha=None, seed=seed):
    
    C = len(set(targets))
    if method.lower()=="iid":
        
        return partition_iid(targets, num_clients, seed)
    
    if method.lower()=="dirichlet":
        
        if alpha is None:
            raise ValueError("Alpha is required for Dirichlet")
        return partition_dirichlet(targets, num_clients, C, alpha, seed)
    
    raise ValueError("method must be 'iid' or 'dirichlet'")

def partition_indices(indices, num_clients, seed=seed):
    
    np.random.seed(seed)
    idx = np.array(indices)
    np.random.shuffle(idx)
    base, rem = divmod(len(idx), num_clients)
    out, st = {}, 0
    
    for c in range(num_clients):
        
        sz = base + (1 if c<rem else 0)
        out[c] = idx[st:st+sz].tolist()
        st += sz
    
    return out

def get_client_loaders(dataset, idx_dict, batch_size, transforms_list=None, shuffle=True):
    
    loaders = []
    
    for c, idx in idx_dict.items():
        
        tf = transforms_list[c] if transforms_list else None
        sub = SubsetWithTransform(dataset, idx, transform=tf)
        loaders.append(DataLoader(sub, batch_size=batch_size, shuffle=shuffle))
    
    return loaders

def get_test_loaders(test_ds, batch_size, num_clients, transforms_list=None, seed=seed):

    idx = list(range(len(test_ds)))
    test_idx = partition_indices(idx, num_clients, seed)
    
    return get_client_loaders(test_ds, test_idx, batch_size, transforms_list, shuffle=False)

def plot_partition_statistics(client_indices_dict, targets, num_classes, dataset_name, method, alpha=None):
    
    data = []
    
    for cid, indices in client_indices_dict.items():
        labels = [targets[i] for i in indices]
        counts = np.bincount(labels, minlength=num_classes)
        data.append(counts)

    data = np.array(data)
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(client_indices_dict))
    for c in range(num_classes):
        ax.barh(range(len(client_indices_dict)), data[:, c], left=bottom, label=f'class{c}')
        bottom += data[:, c]

    ax.set_yticks(range(len(client_indices_dict)))
    ax.set_yticklabels([f'Client {i}' for i in range(len(client_indices_dict))])
    ax.set_xlabel("sample num")
    ax.set_ylabel("client")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    alpha_str = f"_alpha{alpha}" if alpha is not None else ""
    filename = f"partition_{dataset_name}_{method.lower()}{alpha_str}.png"
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

def get_acs_client_loaders(dataset_name: str, n_clients: int, train_batch_size: int, test_batch_size: int, survey_year: str = "2018", horizon: str = "1-Year", survey: str = "person" ):
    
    """
    Returns two lists of length n_clients:
      - train_loaders[i]
      - test_loaders[i]
    Each client fetches ACS data for a specific state, scales, splits 80/20, and wraps in TensorDatasets/DataLoaders.
    """

    # map client_id -> state
    state_map = {
        0: "AL", 1: "AZ", 2: "AR", 3: "CA", 4: "MA", 5: "NC", 6: "TX", 7: "UT", 8: "KS", 9: "KY"
    }

    data_source = ACSDataSource(survey_year=survey_year, horizon=horizon, survey=survey)

    train_loaders = []
    test_loaders  = []

    for cid in range(n_clients):

        # pick state (wrap around if cid >= 10)
        st = state_map[cid % len(state_map)]
        df = data_source.get_data(states=[st], download=True)

        # convert to numpy
        if dataset_name == "acsemployment":
            features, labels, group = ACSEmployment.df_to_numpy(df)
        elif dataset_name == "acsincome":
            features, labels, group = ACSIncome.df_to_numpy(df)
        elif dataset_name == "acspubliccoverage":
            features, labels, group = ACSPublicCoverage.df_to_numpy(df)
        elif dataset_name == "acsmobility":
            features, labels, group = ACSMobility.df_to_numpy(df)
        else:
            raise ValueError(f"Unknown ACS dataset: {dataset_name}")

        # scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        # to tensors
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(labels.astype(float), dtype=torch.float32)

        # dataset + split
        full = TensorDataset(X_t, y_t)
        train_sz = int(0.8 * len(full))
        test_sz  = len(full) - train_sz
        train_ds, test_ds = torch.utils.data.random_split(full, [train_sz, test_sz])

        # loaders
        train_ld = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
        test_ld  = DataLoader(test_ds,  batch_size=test_batch_size,  shuffle=False)

        train_loaders.append(train_ld)
        test_loaders.append(test_ld)

    return train_loaders, test_loaders

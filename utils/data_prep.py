import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset as Dataset_TG
import os
import numpy as np
import math
import pandas as pd


class LabelledDataset(Dataset_TG):
    def __init__(self, npy_file, processed_dir):
        self.npy_ar = np.load(npy_file)
        self.processed_dir = processed_dir

        self.protein_1 = self.npy_ar[:, 2]
        self.protein_2 = self.npy_ar[:, 5]

        self.label = self.npy_ar[:, 6].astype(float)
        self.n_samples = self.npy_ar.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        prot_1_path = os.path.join(self.processed_dir, self.protein_1[index] + ".pt")
        prot_2_path = os.path.join(self.processed_dir, self.protein_2[index] + ".pt")

        print(f"First prot is {prot_1_path}")
        print(f"Second prot is {prot_2_path}")

        prot_1 = torch.load(prot_1_path)
        prot_2 = torch.load(prot_2_path)

        return prot_1, prot_2, torch.tensor(self.label[index])

    def len(self):
        return len(self)

    def get(self, index):
        return self[index]


processed_dir = "../datasets/processed/new_pts/"

ppi_dataset = pd.read_csv("../datasets/processed/ppi_dataset.csv")
npy_ar = np.array(ppi_dataset)

dataset = LabelledDataset(npy_ar, processed_dir)


DATASET_SIZE = len(dataset)
trainset, testset = torch.utils.data.random_split(
    dataset,
    [math.floor(0.8 * DATASET_SIZE), DATASET_SIZE - math.floor(0.8 * DATASET_SIZE)],
)
BATCH_SIZE = 4

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE)
testloader = DataLoader(testset, batch_size=BATCH_SIZE)

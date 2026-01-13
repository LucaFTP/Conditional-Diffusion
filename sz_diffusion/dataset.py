import os
import numpy as np
import pandas as pd

import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from sz_diffusion.gen_utils import dynamic_range_opt
    
class tszDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        redshift_threshold: float,
        col_names: dict = {
            'id':'id','redshift':'redshift', 'mass':'mass',
            'simulation':'simulation', 'snap':'snap', 'ax':'ax', 'rot':'rot'
            },
        epsilon: float = 1e-6,
        mult_factor: float = 1.0,
        transform=None,
    ):
        """
        Args:
            data_dir: Cartella contenente i file .npy.
            col_names: Dizionario per mappare i nomi delle colonne del CSV.
        """
        self.data_dir = os.path.join(root_dir, "npy_files")
        self.csv_path = os.path.join(root_dir, "mainframe.csv")
        self.transform = transform
        self.epsilon = epsilon
        self.mult_factor = mult_factor
        self.col_names = col_names

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            
        df = pd.read_csv(self.csv_path)
        
        self.meta_data = df[df["redshift"] <= redshift_threshold].reset_index(drop=True)
        
        if len(self.meta_data) == 0:
            print(f"Warning: No data found with redshift <= {redshift_threshold}")

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):

        row = self.meta_data.iloc[idx]
        
        file_id = str(row[self.col_names["id"]])
        npy_path = os.path.join(self.data_dir, f"{file_id}.npy")
        
        image = np.load(npy_path).astype(np.float32)
        image = dynamic_range_opt(
            image, epsilon=self.epsilon, mult_factor=self.mult_factor
        )

        if self.transform:
            image = self.transform(image)

        mass_val = float(row[self.col_names["mass"]])
        mass_tensor = torch.tensor(10**(mass_val - 13.8), dtype=torch.float32)

        return image, mass_tensor

# Function to load the dataset and create the DataLoader
def create_dataloader(
        root_dir, dataset_config: dict
        ) -> tuple[tszDataset, DistributedSampler, DataLoader]:
    
    batch_size = dataset_config.get("batch_size", 32)
    img_size = dataset_config.get("img_size", 128)
    augment_horizontal_flip = dataset_config.get("augment_horizontal_flip", True)
    num_workers = len(os.sched_getaffinity(0))

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5 if augment_horizontal_flip else 0.0),
        ]
    )

    dataset = tszDataset(
        root_dir=root_dir,
        redshift_threshold=dataset_config.get("reds_threshold", 1.0),
        epsilon=dataset_config.get("epsilon", 1e-6),
        mult_factor=dataset_config.get("mult_factor", 1.0),
        transform=transform
    )

    # Sampler for DDP
    sampler = DistributedSampler(dataset) if torch.distributed.is_initialized() else None

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True
    )

    return dataset, sampler, train_loader

if __name__ == "__main__":
    dataset, sampler, train_loader = create_dataloader(
        root_dir="/leonardo_scratch/fast/uTS25_Fontana/redshift_zero_folder/",
        dataset_config={}
    )
    print(f"Number of batches: {len(train_loader)}")
    for images, masses in train_loader:
        print(f"Images shape: {images.shape}")
        print(f"Mass condition: {masses.shape}")
        print("Mass values (first batch):", masses)
        break
import os
import numpy as np

import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from gen_utils import dynamic_range_opt
    
class NpyDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            reds_threshold: float,
            epsilon: float = 1e-6,
            mult_factor: float = 1,
            transform=None,
        ):
        
        self.root_dir = root_dir
        self.transform = transform;     self.reds_threshold = reds_threshold
        self.epsilon = epsilon;         self.mult_factor = mult_factor

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"The directory {root_dir} does not exist.")
        
        # Filtra i file npy in base al redshift
        self.npy_files = []
        for f in os.listdir(root_dir):
            reds = float(f.split("reds=")[1][0:4])
            if 0.2 < reds <= self.reds_threshold:
                self.npy_files.append(f)

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.npy_files[idx])
        image = np.load(img_name)
        image = image[362:662, 332:662]
        image = dynamic_range_opt(image, epsilon=self.epsilon, mult_factor=self.mult_factor)

        if self.transform:
            image = self.transform(image)

        mass = float(img_name.split("mass=")[1][0:4])
        mass_tensor = torch.tensor(10**(mass - 13.8), dtype=torch.float32)

        return image, mass_tensor
      
# Funzione per caricare il dataset e creare il DataLoader
def create_dataloader(
        root_dir, dataset_config: dict
        ) -> tuple[NpyDataset, DistributedSampler, DataLoader]:
    
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

    dataset = NpyDataset(
        root_dir=root_dir,
        reds_threshold=dataset_config.get("reds_threshold", 1.0),
        epsilon=dataset_config.get("epsilon", 1e-6),
        mult_factor=dataset_config.get("mult_factor", 1.0),
        transform=transform
    )

    # Sampler per DDP
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
        root_dir="/leonardo_scratch/fast/uTS25_Fontana/ALL_ROT_npy_version/1024x1024/",
        dataset_config={}
    )
    print(f"Number of batches: {len(train_loader)}")
    for images, masses in train_loader:
        print(f"Images shape: {images.shape}")
        print(f"Mass condition: {masses.shape}")
        print("Mass values (first batch):", masses)
        break
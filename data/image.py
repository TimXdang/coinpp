import os
from typing import Callable, Optional
from PIL import Image
import imageio
import requests
import torch
import torchvision
import zipfile

from pathlib import Path
from typing import Any, Callable, Optional
from torchvision.transforms import ToTensor
from torchvision.datasets import VisionDataset


class CIFAR10(torchvision.datasets.CIFAR10):
    """CIFAR10 dataset without labels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.data[index])
        else:
            return self.data[index]


class MNIST(torchvision.datasets.MNIST):
    """MNIST dataset without labels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        # Add channel dimension, convert to float and normalize to [0, 1]
        datapoint = self.data[index].unsqueeze(0).float() / 255.0
        if self.transform:
            return self.transform(datapoint)
        else:
            return datapoint


class Kodak(torch.utils.data.Dataset):
    """Kodak dataset."""

    base_url = "http://r0k.us/graphics/kodak/kodak/"
    num_images = 24
    width = 768
    height = 512
    resolution_hw = (height, width)

    def __init__(
        self,
        root: Path = Path.cwd() / "kodak-dataset",
        transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self.root = root

        self.transform = transform

        if download:
            self.download()

        self.data = tuple(
            imageio.imread(self.root / f"kodim{idx + 1:02}.png")
            for idx in range(self.num_images)
        )

    def _check_exists(self) -> bool:
        # This can be obviously be improved for instance by comparing checksums.
        return (
            self.root.exists() and len(list(self.root.glob("*.png"))) == self.num_images
        )

    def download(self):
        if self._check_exists():
            return

        self.root.mkdir(parents=True, exist_ok=True)

        print(f"Downloading Kodak dataset to {self.root}...")

        for idx in range(self.num_images):
            filename = f"kodim{idx + 1:02}.png"
            with open(self.root / filename, "wb") as f:
                f.write(
                    requests.get(
                        f"http://r0k.us/graphics/kodak/kodak/{filename}"
                    ).content
                )

        print("Done!")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Any:
        image = self.data[index]
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __repr__(self) -> str:
        head = "Dataset Kodak"
        body = []
        body.append(f"Number of images: {self.num_images}")
        body.append(f"Root location: {self.root}")
        lines = [head] + ["    " + line for line in body]
        return "\n".join(lines)


class SliceInfo:
    def __init__(self, img, patient, modality, img_slice):
        self.img = img
        self.patient = patient
        self.modality = modality
        self.img_slice = img_slice


class BraTSGLIDataset(VisionDataset):

    def __init__(self, root, transforms=None, transform=None, target_transform=None, split='train', modality='all') -> None:
        super().__init__(root, transforms, transform, target_transform)

        assert modality in ['all', 't1c', 't1n', 't2f', 't2w'], 'The requested modality is not valid.'

        self.all_data = self._load_data()
        self.num_patients = len(set([d.patient for d in self.all_data]))
        if modality == 'all':
            self.n_imgs = self.num_patients
            data = self.all_data
        elif modality == 't1c':
            data = [d for d in self.all_data if d.modality == 't1c']
            self.n_imgs = len(set([d.patient for d in data]))
        elif modality == 't1n':
            data = [d for d in self.all_data if d.modality == 't1n']
            self.n_imgs = len(set([d.patient for d in data]))
        elif modality == 't2f':
            data = [d for d in self.all_data if d.modality == 't2f']
            self.n_imgs = len(set([d.patient for d in data]))
        elif modality == 't2w':
            data = [d for d in self.all_data if d.modality == 't2w']
            self.n_imgs = len(set([d.patient for d in data]))

        if split == 'train':
            split_idx = int(0.8 * self.n_imgs)
            patients = sorted(set(d.patient for d in data))[:split_idx]
            self.data = [d for d in data if d.patient in patients]
        elif split == 'val':
            split_idx = int(0.8 * self.n_imgs)
            patients = sorted(set(d.patient for d in data))[split_idx:]
            self.data = [d for d in data if d.patient in patients]

    def _load_data(self):
        img_folder = self.root
        img_slices = []
        for fname in sorted(os.listdir(img_folder)):
            slice_info = SliceInfo(img=os.path.join(img_folder, fname), patient='-'.join(fname.split('-')[2:4]), modality=fname[20:23],
                                   img_slice=fname[23:-5])
            img_slices.append(slice_info)
        return img_slices
    
    def __getitem__(self, index: int):
        with Image.open(self.data[index].img) as img:
            # Convert PIL Image to tensor
            #img = ToTensor()(img)
        
            if self.transform is not None:
                img = self.transform(img)

        return img
    
    def __len__(self) -> int:
        return len(self.data)

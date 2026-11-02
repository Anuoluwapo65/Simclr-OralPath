import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from pathlib import Path 
import torch.utils.data as data 


class TileLoader(data.Dataset):
    def __init__(self, root_dir,transform = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_files = [
            p for p in self.root_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = self.load_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image

    def load_image(self, img_path):
        from PIL import Image 
        img = Image.open(img_path)
        img = img.convert("RGB")
        return img 



class ContrastiveLearningViewGenerator(object):
    def __init__(self, base_dataset, n_views = 2, transform = None):
        self.base_dataset = base_dataset
        self.n_views = n_views 
        self.transform = transform 


    def __len__(self):
        return len(self.base_dataset)


    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        if self.transform:
            views = [self.transform(sample) for _ in range(self.n_views)]
        else:
            views = [sample] * self.n_views
        # The training loop unpacks each batch as (views, _).
        return views, idx




    

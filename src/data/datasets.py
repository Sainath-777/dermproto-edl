import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BaseSkinDataset(Dataset):
    """
    Abstract base class for dermatological datasets.
    """
    def __init__(self, root: str, split_classes: list, transform=None, mode: str = 'train', verify_distribution: bool = True):
        self.root = root
        self.split_classes = split_classes
        self.transform = transform
        self.mode = mode
        
        # Load and filter raw metadata (to be defined by subclass)
        raw_df = self.load_metadata()
        
        # Filter for active split classes
        self.df = raw_df[raw_df["label_str"].isin(self.split_classes)].reset_index(drop=True)
        self.class_names = sorted(list(self.df["label_str"].unique()))
        
        # Mode-based train/val split: 80% train, 20% validation, sorted deterministically by path
        if self.mode in ('train', 'val'):
            split_dfs = []
            for class_name in self.class_names:
                class_df = self.df[self.df["label_str"] == class_name].copy()
                class_df = class_df.sort_values(by="image_path").reset_index(drop=True)
                n = len(class_df)
                split_idx = int(n * 0.8)
                if self.mode == 'train':
                    split_dfs.append(class_df.iloc[:split_idx])
                else:  # val
                    split_dfs.append(class_df.iloc[split_idx:])
            self.df = pd.concat(split_dfs).reset_index(drop=True)
            self.class_names = sorted(list(self.df["label_str"].unique()))
        elif self.mode == 'all':
            pass  # Use all available images, no train/val split (for OOD evaluation)
                
        # Define class mapping
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.df["label"] = self.df["label_str"].map(self.class_to_idx)
        
        # Group indices by class for fast episodic sampling
        self.class_indices = {
            c: self.df[self.df["label_str"] == c].index.tolist() 
            for c in self.class_names
        }
        
        if verify_distribution:
            self.print_distribution()
            
    def load_metadata(self) -> pd.DataFrame:
        """
        Loads metadata. Must return a DataFrame with columns: ['image_path', 'label_str']
        """
        raise NotImplementedError("Subclasses must implement load_metadata()")
        
    def print_distribution(self):
        print(f"\n--- Dataset Class Distribution ({self.__class__.__name__} - Mode: {self.mode}) ---")
        counts = self.df["label_str"].value_counts()
        for name in self.class_names:
            count = counts.get(name, 0)
            print(f"Class {name:20s}: {count:5d} images")
        print(f"Total filtered images: {len(self.df)}")
        print("---------------------------------------------------\n")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label = row["label"]
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")
            
        # Read image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.tensor(image_np.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            
        return image_tensor, label


class HAM10000Dataset(BaseSkinDataset):
    """
    Dataset wrapper for HAM10000 dataset.
    """
    def __init__(self, root: str, split_classes: list, transform=None, mode: str = 'train', verify_distribution: bool = True):
        super().__init__(root, split_classes, transform, mode, verify_distribution)

    def load_metadata(self) -> pd.DataFrame:
        csv_path = os.path.join(self.root, "HAM10000_metadata.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
        
        image_paths = []
        for _, row in df.iterrows():
            image_id = row["image_id"]
            found = False
            for part_folder in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
                path = os.path.join(self.root, part_folder, f"{image_id}.jpg")
                if os.path.exists(path):
                    image_paths.append(path)
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"Image {image_id}.jpg not found in part 1 or part 2 folders of {self.root}")
                
        df["image_path"] = image_paths
        df["label_str"] = df["dx"]
        return df[["image_path", "label_str"]]


class ISIC2019Dataset(BaseSkinDataset):
    """
    Dataset wrapper for ISIC 2019. Decodes one-hot ground truth CSV.
    """
    ISIC2019_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

    def __init__(self, root: str, split_classes: list = None, transform=None, mode: str = 'all', verify_distribution: bool = False):
        if split_classes is None:
            split_classes = self.ISIC2019_CLASSES
        super().__init__(root, split_classes, transform, mode, verify_distribution)

    def load_metadata(self) -> pd.DataFrame:
        csv_path = os.path.join(self.root, "ISIC_2019_Training_GroundTruth.csv")
        if not os.path.exists(csv_path):
            alt_csv = os.path.join(self.root, "train.csv")
            if os.path.exists(alt_csv):
                csv_path = alt_csv
            else:
                raise FileNotFoundError(f"ISIC 2019 metadata CSV not found in {self.root}")
            
        df = pd.read_csv(csv_path)
        class_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
        present_cols = [c for c in class_cols if c in df.columns]
        
        # Decode one-hot columns via argmax
        df["label_str"] = df[present_cols].idxmax(axis=1)
        df = df[df["label_str"] != "UNK"].reset_index(drop=True)
        
        image_col = "image" if "image" in df.columns else df.columns[0]
        img_dir_1 = os.path.join(self.root, "ISIC_2019_Training_Input")
        img_dir_2 = os.path.join(self.root, "train")
        
        image_paths = []
        for _, row in df.iterrows():
            img_name = f"{row[image_col]}.jpg" if not str(row[image_col]).endswith(".jpg") else row[image_col]
            path1 = os.path.join(img_dir_1, img_name)
            path2 = os.path.join(img_dir_2, img_name)
            path_root = os.path.join(self.root, img_name)
            
            if os.path.exists(path1):
                image_paths.append(path1)
            elif os.path.exists(path2):
                image_paths.append(path2)
            elif os.path.exists(path_root):
                image_paths.append(path_root)
            else:
                raise FileNotFoundError(f"ISIC 2019 image {img_name} not found in {self.root}")
                
        df["image_path"] = image_paths
        return df[["image_path", "label_str"]]


class SD198Dataset(BaseSkinDataset):
    """
    Dataset wrapper for SD-198. Discovers class names from directory structure (no CSV).
    """
    def __init__(self, root: str, split_classes: list = None, transform=None, mode: str = 'all', verify_distribution: bool = False, min_images_per_class: int = 10):
        self.min_images_per_class = min_images_per_class
        if split_classes is None:
            split_classes = self._discover_valid_classes(root, min_images_per_class)
        super().__init__(root, split_classes, transform, mode, verify_distribution)

    def _discover_valid_classes(self, root: str, min_count: int) -> list:
        valid = []
        if not os.path.isdir(root):
            raise FileNotFoundError(f"SD-198 root directory not found: {root}")
        for class_name in sorted(os.listdir(root)):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            imgs = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(imgs) >= min_count:
                valid.append(class_name)
        print(f"SD-198: Found {len(valid)} classes with >= {min_count} images.")
        return valid

    def load_metadata(self) -> pd.DataFrame:
        records = []
        for class_name in sorted(os.listdir(self.root)):
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    records.append({
                        "image_path": os.path.join(class_dir, fname),
                        "label_str": class_name
                    })
        return pd.DataFrame(records)


def build_transforms(mode: str) -> A.Compose:
    if mode == "train":
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:  # val or test
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
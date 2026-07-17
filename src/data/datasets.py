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
            print(f"Class {name:5s}: {count:5d} images")
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
            # Default fallback to tensor conversion if no transforms provided
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
        
        # Locate image paths across the two subfolders
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


def build_transforms(mode: str) -> A.Compose:
    """
    Build transform pipeline using Albumentations.
    Uses standard DINOv2 / ImageNet normalization parameters.
    """
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


def generate_mock_dataset(root_path: str):
    """
    Generates a small mock dataset locally for testing/compilation checks.
    """
    os.makedirs(root_path, exist_ok=True)
    img_dir_1 = os.path.join(root_path, "HAM10000_images_part_1")
    img_dir_2 = os.path.join(root_path, "HAM10000_images_part_2")
    os.makedirs(img_dir_1, exist_ok=True)
    os.makedirs(img_dir_2, exist_ok=True)
    
    csv_path = os.path.join(root_path, "HAM10000_metadata.csv")
    if not os.path.exists(csv_path):
        print("Generating mock HAM10000 metadata and images locally...")
        classes = ["nv", "mel", "bkl", "bcc", "akiec", "df", "vasc"]
        data = []
        for i in range(100):
            image_id = f"ISIC_00{24306 + i}"
            lesion_id = f"HAM_00{10000 + i // 2}"
            dx = classes[i % len(classes)]
            target_dir = img_dir_1 if i % 2 == 0 else img_dir_2
            img_path = os.path.join(target_dir, f"{image_id}.jpg")
            
            # Create a dummy image (100x100 RGB color block)
            if not os.path.exists(img_path):
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                Image.fromarray(img).save(img_path)
                
            data.append({
                "lesion_id": lesion_id,
                "image_id": image_id,
                "dx": dx,
                "dx_type": "consensus",
                "age": 50.0,
                "sex": "male",
                "localization": "back"
            })
            
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"Mock dataset generated successfully at: {root_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock_dir", type=str, default="data/raw")
    args = parser.parse_args()
    
    generate_mock_dataset(args.mock_dir)
    print("Mock generation check completed!")
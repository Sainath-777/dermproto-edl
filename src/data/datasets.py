import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def find_csv_metadata(root: str, required_cols: list = None, keywords: list = None) -> str:
    """
    Recursively searches `root` for a metadata CSV file matching keywords or required columns.
    """
    if keywords is None:
        keywords = ["groundtruth", "train", "metadata", "isic"]
    
    candidates = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(".csv"):
                candidates.append(os.path.join(dirpath, fname))
                
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in dataset root directory: {root}")
        
    matching_keyword_candidates = []
    for cand in candidates:
        cand_name = os.path.basename(cand).lower()
        if any(kw in cand_name for kw in keywords):
            matching_keyword_candidates.append(cand)
            
    pool = matching_keyword_candidates if matching_keyword_candidates else candidates
    
    if required_cols:
        for cand in pool:
            try:
                header_df = pd.read_csv(cand, nrows=1)
                if any(col in header_df.columns for col in required_cols):
                    return cand
            except Exception:
                continue
                
    if pool:
        return pool[0]
    raise FileNotFoundError(f"Could not find any suitable metadata CSV in {root}")


def build_image_map(root: str, valid_exts: tuple = (".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG")) -> dict:
    image_map = {}
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(valid_exts):
                full_path = os.path.join(dirpath, fname)
                image_map[fname] = full_path
                stem = os.path.splitext(fname)[0]
                if stem not in image_map:
                    image_map[stem] = full_path
    return image_map


class BaseSkinDataset(Dataset):
    """
    Abstract base class for dermatological datasets.
    """
    def __init__(self, root: str, split_classes: list, transform=None, mode: str = 'train', verify_distribution: bool = True):
        self.root = root
        self.split_classes = split_classes
        self.transform = transform
        self.mode = mode
        
        raw_df = self.load_metadata()
        
        self.df = raw_df[raw_df["label_str"].isin(self.split_classes)].reset_index(drop=True)
        self.class_names = sorted(list(self.df["label_str"].unique()))
        
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
            pass  # Use all available images
                
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.df["label"] = self.df["label_str"].map(self.class_to_idx)
        
        self.class_indices = {
            c: self.df[self.df["label_str"] == c].index.tolist() 
            for c in self.class_names
        }
        
        if verify_distribution:
            self.print_distribution()
            
    def load_metadata(self) -> pd.DataFrame:
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
            
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.tensor(image_np.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            
        return image_tensor, label


class HAM10000Dataset(BaseSkinDataset):
    def __init__(self, root: str, split_classes: list, transform=None, mode: str = 'train', verify_distribution: bool = True):
        super().__init__(root, split_classes, transform, mode, verify_distribution)

    def load_metadata(self) -> pd.DataFrame:
        csv_path = find_csv_metadata(
            self.root,
            required_cols=["image_id", "dx"],
            keywords=["ham10000", "metadata"]
        )
        df = pd.read_csv(csv_path)
        
        image_map = build_image_map(self.root)
        image_paths = []
        matched_mask = []
        for _, row in df.iterrows():
            img_id = str(row["image_id"]).strip()
            path = image_map.get(img_id) or image_map.get(f"{img_id}.jpg") or image_map.get(os.path.splitext(img_id)[0])
            if path:
                image_paths.append(path)
                matched_mask.append(True)
            else:
                matched_mask.append(False)
                
        if not any(matched_mask):
            raise FileNotFoundError(f"Could not resolve any HAM10000 image files in {self.root} using CSV {csv_path}")
        elif not all(matched_mask):
            print(f"Warning: {len(matched_mask) - sum(matched_mask)} images out of {len(df)} could not be located in {self.root}.")
            df = df[matched_mask].reset_index(drop=True)
                
        df["image_path"] = image_paths
        df["label_str"] = df["dx"]
        return df[["image_path", "label_str"]]


class ISIC2019Dataset(BaseSkinDataset):
    """
    Dataset wrapper for ISIC 2019. Supports CSV ground-truth OR folder-per-class format.
    """
    ISIC2019_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

    def __init__(self, root: str, split_classes: list = None, transform=None, mode: str = 'all', verify_distribution: bool = False):
        if split_classes is None:
            split_classes = self.ISIC2019_CLASSES
        super().__init__(root, split_classes, transform, mode, verify_distribution)

    def load_metadata(self) -> pd.DataFrame:
        # Check if CSV metadata exists
        try:
            csv_path = find_csv_metadata(
                self.root, 
                required_cols=["MEL", "NV", "BCC"],
                keywords=["groundtruth", "train", "isic"]
            )
            df = pd.read_csv(csv_path)
            class_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
            present_cols = [c for c in class_cols if c in df.columns]
            
            df["label_str"] = df[present_cols].idxmax(axis=1)
            df = df[df["label_str"] != "UNK"].reset_index(drop=True)
            
            image_col = "image" if "image" in df.columns else df.columns[0]
            image_map = build_image_map(self.root)
            
            image_paths = []
            matched_mask = []
            for _, row in df.iterrows():
                img_id = str(row[image_col]).strip()
                path = image_map.get(img_id) or image_map.get(os.path.splitext(img_id)[0])
                if path:
                    image_paths.append(path)
                    matched_mask.append(True)
                else:
                    matched_mask.append(False)
                    
            if any(matched_mask):
                df = df[matched_mask].reset_index(drop=True)
                df["image_path"] = image_paths
                return df[["image_path", "label_str"]]
        except FileNotFoundError:
            pass

        # Fallback: Folder-per-class structure (e.g. root/MEL/*.jpg, root/NV/*.jpg)
        records = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            folder_name = os.path.basename(dirpath)
            if folder_name in self.ISIC2019_CLASSES:
                for fname in filenames:
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        records.append({
                            "image_path": os.path.join(dirpath, fname),
                            "label_str": folder_name
                        })
        if records:
            return pd.DataFrame(records)
            
        raise FileNotFoundError(f"Could not load ISIC 2019 metadata via CSV or class subfolders in {self.root}")


class SD198Dataset(BaseSkinDataset):
    """
    Dataset wrapper for SD-198. Discovers class names dynamically across directory structures.
    """
    def __init__(self, root: str, split_classes: list = None, transform=None, mode: str = 'all', verify_distribution: bool = False, min_images_per_class: int = 10):
        self.min_images_per_class = min_images_per_class
        self.effective_root = self._find_effective_root(root)
        if split_classes is None:
            split_classes = self._discover_valid_classes(self.effective_root, min_images_per_class)
        super().__init__(root, split_classes, transform, mode, verify_distribution)

    def _find_effective_root(self, root: str) -> str:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"SD-198 root directory not found: {root}")
            
        curr = root
        while True:
            subdirs = [d for d in os.listdir(curr) if os.path.isdir(os.path.join(curr, d))]
            direct_images = False
            for d in subdirs:
                d_path = os.path.join(curr, d)
                files = os.listdir(d_path)
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files if os.path.isfile(os.path.join(d_path, f))):
                    direct_images = True
                    break
            if direct_images:
                return curr
            if len(subdirs) == 1:
                curr = os.path.join(curr, subdirs[0])
            else:
                break
        return curr

    def _discover_valid_classes(self, root: str, min_count: int) -> list:
        valid = []
        for class_name in sorted(os.listdir(root)):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            imgs = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(imgs) >= min_count:
                valid.append(class_name)
        print(f"SD-198: Found {len(valid)} classes with >= {min_count} images in {root}.")
        return valid

    def load_metadata(self) -> pd.DataFrame:
        eff_root = getattr(self, 'effective_root', self._find_effective_root(self.root))
        records = []
        for class_name in sorted(os.listdir(eff_root)):
            class_dir = os.path.join(eff_root, class_name)
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
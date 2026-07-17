import random
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

class EpisodeSampler:
    """
    Episodic sampler for few-shot learning tasks.
    Samples K-way, N-shot, Q-query episodes from a BaseSkinDataset.
    """
    def __init__(self, dataset, k_way: int, n_shot: int, n_query: int):
        self.dataset = dataset
        self.k_way = k_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.classes = dataset.class_names

    def sample_episode(self) -> dict:
        """
        Samples a single episode: K classes, N support images, Q query images.
        Returns a dict of tensors mapped to local labels [0, K-1].
        """
        if len(self.classes) < self.k_way:
            raise ValueError(
                f"Dataset only has {len(self.classes)} classes, but requested k_way is {self.k_way}"
            )
            
        # 1. Randomly sample K classes
        selected_classes = random.sample(self.classes, self.k_way)
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        # 2. Local label mapping dictionary (maps selected classes to 0 to K-1)
        local_label_map = {class_name: i for i, class_name in enumerate(selected_classes)}
        
        for class_name in selected_classes:
            class_indices = self.dataset.class_indices[class_name]
            needed_samples = self.n_shot + self.n_query
            
            # Guard: sample with replacement if category doesn't contain enough images
            if len(class_indices) < needed_samples:
                print(f"Warning: Class {class_name} has only {len(class_indices)} images. "
                      f"Requested {needed_samples}. Sampling with replacement.")
                sampled_indices = random.choices(class_indices, k=needed_samples)
            else:
                sampled_indices = random.sample(class_indices, k=needed_samples)
                
            # Split indices into support and query
            support_idxs = sampled_indices[:self.n_shot]
            query_idxs = sampled_indices[self.n_shot:]
            
            # Local label index
            local_label = local_label_map[class_name]
            
            # Load images and append local labels
            for idx in support_idxs:
                img_tensor, _ = self.dataset[idx]
                support_images.append(img_tensor)
                support_labels.append(local_label)
                
            for idx in query_idxs:
                img_tensor, _ = self.dataset[idx]
                query_images.append(img_tensor)
                query_labels.append(local_label)
                
        # Stack all lists into single tensors
        return {
            "support_images": torch.stack(support_images),            # Shape: (K * N, C, H, W)
            "support_labels": torch.tensor(support_labels, dtype=torch.long), # Shape: (K * N)
            "query_images": torch.stack(query_images),                # Shape: (K * Q, C, H, W)
            "query_labels": torch.tensor(query_labels, dtype=torch.long),     # Shape: (K * Q)
            "class_names": selected_classes
        }


def plot_episode(episode: dict, save_path: str):
    """
    Plots sampled support and query sets in a grid and saves to disk.
    De-normalizes standard ImageNet normalization to show correct RGB colors.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Normalization mean/std to revert
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    k_way = len(episode["class_names"])
    n_shot = len(episode["support_images"]) // k_way
    n_query = len(episode["query_images"]) // k_way
    
    fig, axes = plt.subplots(k_way, n_shot + n_query, figsize=(3 * (n_shot + n_query), 3 * k_way))
    fig.suptitle("Sampled Episode (Left: Support, Right: Query)", fontsize=16, weight="bold")
    
    for row in range(k_way):
        class_name = episode["class_names"][row]
        
        # 1. Plot Support Images
        for col in range(n_shot):
            idx = row * n_shot + col
            img = episode["support_images"][idx].numpy().transpose(1, 2, 0)
            img = (img * std + mean).clip(0, 1) # De-normalize
            
            ax = axes[row, col] if k_way > 1 else axes[col]
            ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(class_name, fontsize=12, weight="bold")
            if row == 0:
                ax.set_title(f"Support {col+1}")
                
        # 2. Plot Query Images
        for col in range(n_query):
            idx = row * n_query + col
            img = episode["query_images"][idx].numpy().transpose(1, 2, 0)
            img = (img * std + mean).clip(0, 1) # De-normalize
            
            ax = axes[row, n_shot + col] if k_way > 1 else axes[n_shot + col]
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Query {col+1}")
                
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Visualized episode grid saved to: {save_path}")


if __name__ == "__main__":
    import yaml
    import sys
    
    # Add parent directory to sys.path so we can import utils.py when running from root
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
        
    from utils import resolve_data_root
    from datasets import HAM10000Dataset, build_transforms, generate_mock_dataset
    
    # Load config
    config_path = "configs/base.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Resolve the data root (uses Kaggle path if present, else local)
    data_root = resolve_data_root(config)
    
    # If the resolved path doesn't exist, generate local mock data as a fallback
    if not os.path.exists(data_root):
        generate_mock_dataset(data_root)
        
    # Initialize dataset (real or mock depending on path resolution)
    dataset = HAM10000Dataset(
        root=data_root,
        split_classes=config["dataset"]["meta_train_classes"],
        transform=build_transforms("train")
    )
    
    # Initialize episodic sampler
    sampler = EpisodeSampler(
        dataset=dataset,
        k_way=config["episode"]["k_way"],
        n_shot=config["episode"]["n_shot"],
        n_query=2 # Sample just 2 query images for quick plotting
    )
    
    episode = sampler.sample_episode()
    print("\n--- Episode Sampled Successfully ---")
    print("Class names picked: ", episode["class_names"])
    print("Support images shape: ", episode["support_images"].shape)
    print("Support labels values: ", episode["support_labels"].tolist())
    print("Query images shape: ", episode["query_images"].shape)
    print("Query labels values: ", episode["query_labels"].tolist())
    
    plot_episode(episode, "experiments/sample_episode.png")
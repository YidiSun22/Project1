import os
import torch

from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from collections import defaultdict


class MultiLabelDataset(Dataset):
    def __init__(self, dataset_type, root_dir="F:/dissertation_project/dataset/DataSet/Experiment_DataSet/dataset_split", transform=None):
        """
        Dataset for multi-label classification, where each image may belong to multiple categories.

        dataset_type: "train_val" or "test"
        root_dir: Root directory of the dataset (default: dataset_split/)
        transform: PyTorch data augmentation transformations
        """
        self.root_dir = os.path.join(root_dir, dataset_type)  # train_val/ or test/
        self.transform = transform
        self.image_paths = []   # Store all image paths
        self.labels = []        # Store corresponding multi-labels
        self.class_to_idx = {}  # Mapping from class names to indices
        self.image_label_map = defaultdict(set)  # Record multiple labels for each image

        # 1️⃣ Get class list and create class index mapping
        class_folders = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_folders)}

        # 2️⃣ Traverse all class folders and record multi-labels for each image
        for cls_name in self.class_to_idx:
            cls_folder = os.path.join(self.root_dir, cls_name)
            for file_name in os.listdir(cls_folder):
                file_path = os.path.join(cls_folder, file_name)
                if os.path.isfile(file_path):  # Ensure the file exists
                    self.image_label_map[file_name].add(cls_name)  # Record the class of this image

        # 3️⃣ Generate final image paths & label list
        for file_name, label_names in self.image_label_map.items():
            # Choose the first class directory of this image to construct the path
            file_path = os.path.join(self.root_dir, list(label_names)[0], file_name)
            if os.path.exists(file_path):  
                self.image_paths.append(file_path)
                # Generate one-hot vector, setting the belonging class index to 1.0
                label_vector = np.zeros(len(self.class_to_idx), dtype=np.float32)
                for lbl in label_names:
                    label_vector[self.class_to_idx[lbl]] = 1.0  
                self.labels.append(label_vector)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")  # Load image
        except Exception as e:
            print(f"⚠️ Unable to load image: {self.image_paths[idx]}, Error: {e}")
            return None, None  # Return None if reading fails

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

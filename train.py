import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torchvision import transforms
import platform
from Dataset import MultiLabelDataset
from Model import CheXNetFPNModel

def train_and_evaluate(dataset, model, device, num_epochs=50, batch_size=32, save_root="./checkpoint"):
    """
    Train a model using an 80/20 training/validation split, compute AUC, 
    and save the latest and best models at each epoch.
    """
    # Training settings
    total_epochs = num_epochs
    eta_max = 3e-2  # Maximum learning rate
    eta_min = 1e-6  # Minimum learning rate

    # Split dataset into training (80%) and validation (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Ensure checkpoint directory exists
    os.makedirs(save_root, exist_ok=True)
    latest_checkpoint = os.path.join(save_root, "latest_checkpoint.pth")
    best_checkpoint = os.path.join(save_root, "best_model.pth")

    # Initialize optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=eta_max, momentum=0.9, weight_decay=1e-4)
    criterion = nn.BCELoss()

    # Define learning rate schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)

    # Training history tracking
    start_epoch = 1
    best_val_auc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}
    lr_values = []

    if os.path.exists(latest_checkpoint):
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_auc = checkpoint["best_val_auc"]
        history = checkpoint["history"]
        print(f"🔄 Resuming training from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, total_epochs + 1):
        print(f"\n🔥 Epoch {epoch}/{total_epochs}")
        model.train()
        running_loss = 0.0
        all_labels, all_preds = [], []

        for inputs, labels in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{total_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_labels.append(labels)
            all_preds.append(outputs.detach())

        # Update learning rate scheduler
        scheduler.step()

        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
        all_preds = torch.cat(all_preds, dim=0).cpu().numpy()

        train_loss = running_loss / len(train_loader)
        macro_auc = np.mean([roc_auc_score(all_labels[:, i], all_preds[:, i]) for i in range(all_labels.shape[1])])
        history["train_loss"].append(train_loss)
        history["train_auc"].append(macro_auc)
        print(f"Train Loss: {train_loss:.6f} | Train AUC: {macro_auc:.6f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"[Val] Epoch {epoch}/{total_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                all_labels.append(labels)
                all_preds.append(outputs)

        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
        all_preds = torch.cat(all_preds, dim=0).cpu().numpy()

        val_loss = val_loss / len(val_loader)
        macro_auc = np.mean([roc_auc_score(all_labels[:, i], all_preds[:, i]) for i in range(all_labels.shape[1])])
        history["val_loss"].append(val_loss)
        history["val_auc"].append(macro_auc)
        print(f"Validation Loss: {val_loss:.6f} | Validation AUC: {macro_auc:.6f}")



        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_auc": best_val_auc,
            "history": history
        }, latest_checkpoint)

        # Save best model if AUC improves
        if macro_auc > best_val_auc:
            best_val_auc = macro_auc
            torch.save(model.state_dict(), best_checkpoint)
            print(f"🏆 New best model saved: {best_checkpoint}")

        # Record and print current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        lr_values.append(current_lr)
        print(f"📉 Current Learning Rate: {current_lr:.6f}")




# ========== #
# Data Augmentation #
# ========== #
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========== #
# Load Data #
# ========== #
num_workers = 4 if platform.system() != "Windows" else 0  # Adapt for Windows
print("num_workers: ", num_workers)

# Training dataset
train_val_dataset = MultiLabelDataset("train_val", transform=data_transforms)
# train_val_loader = DataLoader(train_val_dataset, batch_size=32, shuffle=True)

# Test dataset
test_dataset = MultiLabelDataset("test", transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32,num_workers=num_workers, shuffle=False)

# Check dataset sizes
print(f"✅ Training + Validation dataset size: {len(train_val_dataset)}")
print(f"✅ Test dataset size: {len(test_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CheXNetFPNModel(num_classes=15).to(device)
print(device)

    
train_and_evaluate(
    dataset=train_val_dataset,  
    model=model,
    device=device,
    num_epochs=50,
    batch_size=32,
    save_root="./checkpoint"
)






import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
import pandas as pd
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    accuracy_score, average_precision_score
)

# Dataset

class ImageNPZDataset(Dataset):
    def __init__(self, npz_file, csv_file, transform=None):

        # load images
        npz_data = np.load(npz_file)
        self.images = npz_data['images'] 
        
        # load labels.
        self.labels = pd.read_csv(csv_file)['label'].values.astype(np.int64)
        
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]

        if not torch.is_tensor(image):
            image = torch.tensor(image, dtype=torch.float)

        if image.ndim == 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# Splitting + Dataloaders

def create_dataloaders(npz_file, csv_file, batch_size=64, seed=42, transform=None):
    full_dataset = ImageNPZDataset(npz_file, csv_file, transform=transform)
    
    torch.manual_seed(seed)
    train_dataset, test_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42, stratify=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader

# Training 

def train_model(model, train_loader, num_epochs, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        scheduler.step()

if __name__ == '__main__':
    # Paths to your data files.
    npz_file = '/path/to/images/'
    csv_file = '/path/to/labels/'
    
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    train_loader, test_loader = create_dataloaders(
        npz_file, csv_file, batch_size=64, transform=transform
    )
    # standard SwinV2 
    num_classes = 8  
    model = timm.create_model(
        'swinv2_tiny_window8_256',  
        pretrained=True,
        num_classes=num_classes,
        img_size=128  
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 100
    
    train_model(model, train_loader, num_epochs=num_epochs, device=device)

    # test

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)[:, 1]  
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    test_acc = accuracy_score(all_labels, all_preds)
    
    test_roc_auc = roc_auc_score(all_labels, all_probs)
    
    test_f1 = f1_score(all_labels, all_preds)
    
    pr_auc = average_precision_score(all_labels, all_probs)
    
    cm = confusion_matrix(all_labels, all_preds)

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) 
    specificity = tn / (tn + fp) 
        
    print(f"Test ROC-AUC    : {test_roc_auc:.4f}")
    print(f"Test Accuracy   : {test_acc:.4f}")
    print(f"Sensitivity     : {sensitivity:.4f}")
    print(f"Specificity     : {specificity:.4f}")
    print(f"Test F1 Score   : {test_f1:.4f}")
    print(f"Test PR-AUC     : {pr_auc:.4f}")


    

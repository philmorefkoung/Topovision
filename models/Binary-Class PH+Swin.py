import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, average_precision_score, confusion_matrix

# 1. TopoRET

class TopoRET(nn.Module):
    def __init__(self, num_classes, img_feat_dim, embed_dim=96):
        super(TopoRET, self).__init__()
        self.embed_dim = embed_dim
        
        self.swin = timm.create_model(
            'swinv2_tiny_window8_256',  
            pretrained=True,
            num_classes=0,
            img_size=128   
        )

        if img_feat_dim != embed_dim:
            self.img_proj = nn.Linear(img_feat_dim, embed_dim)
        else:
            self.img_proj = nn.Identity()
        
        self.PH_embed = nn.Linear(400, embed_dim)

        # cross-attention
        self.cross_attn_img = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.cross_attn_PH = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        
        # self attention
        self.self_attn_img = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.self_attn_PH = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        
        # fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(4 * embed_dim, 3 * embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(3 * embed_dim, 3 * embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.classifier = nn.Linear(3 * embed_dim, num_classes)

    def forward(self, image, PH):

        img_features = self.swin.forward_features(image) 
        B, C, H, W = img_features.shape
        img_tokens = img_features.flatten(2).transpose(1, 2) 
        img_tokens = self.img_proj(img_tokens) 
        
        PH_embedded = self.PH_embed(PH)  # Now shape: (B, embed_dim)
        PH_tokens = PH_embedded.unsqueeze(1)  # Shape: (B, 1, embed_dim)

        # Dual Cross Attention
        img_attended, _ = self.cross_attn_img(query=img_tokens, key=PH_tokens, value=PH_tokens)
        PH_attended, _ = self.cross_attn_PH(query=PH_tokens, key=img_tokens, value=img_tokens)
        
        # Self-Attention Branches
        img_self, _ = self.self_attn_img(query=img_tokens, key=img_tokens, value=img_tokens)
        PH_self, _ = self.self_attn_PH(query=PH_tokens, key=PH_tokens, value=PH_tokens)
        
        # Pooling
    
        img_attended_pooled = img_attended.mean(dim=1)  
        PH_attended_pooled = PH_attended.mean(dim=1)     

        img_self_pooled = img_self.mean(dim=1)             
        PH_self_pooled = PH_self.mean(dim=1)             

        # Fusion MLP
        fused_features = torch.cat([img_attended_pooled, PH_attended_pooled,
                                    img_self_pooled, PH_self_pooled], dim=-1)  

        fused = self.fusion_mlp(fused_features) 
        
        logits = self.classifier(fused)  
        return logits

# Dataset

class ImageNPZPHDataset(Dataset):
    def __init__(self, npz_file, PH_file, transform=None):

        # images
        npz_data = np.load(npz_file)
        self.images = npz_data['images']  
        
        # Betti Vectors
        self.PH_data = pd.read_PH(PH_file)
        
        # extract betti vectors
        feature_cols = [str(i) for i in range(400)]
        self.PH_features = self.PH_data[feature_cols].values.astype(np.float32)
        
        # Extract labels
        self.labels = self.PH_data['label'].values.astype(np.int64)
        
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
        
        PH_feat = torch.tensor(self.PH_features[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, PH_feat, label

# Splitting & Dataloaders

def create_dataloaders(npz_file, PH_file, batch_size=64, seed=42, transform=None):

    full_dataset = ImageNPZPHDataset(npz_file, PH_file, transform=transform)
    
    torch.manual_seed(seed)
    
    train_dataset, test_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

# training 

def train_model(model, train_loader, num_epochs, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, PH_feats, labels in train_loader:
            images = images.to(device)
            PH_feats = PH_feats.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, PH_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        scheduler.step()

if __name__ == '__main__':
    # path to files
    npz_file = '/path/to/npz'
    PH_file = '/path/to/betti_vectors/and/labels'
    
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    train_loader, test_loader = create_dataloaders(
        npz_file, PH_file, batch_size=64, transform=transform
    )
    
    # Create a dummy image to match our input resolution.
    dummy_image = torch.randn(1, 3, 128, 128)
    swin_dummy = timm.create_model(
        'swinv2_tiny_window8_256',
        pretrained=True,
        num_classes=0,
        img_size=128   
    )

    with torch.no_grad():
        features = swin_dummy.forward_features(dummy_image)
    _, C, _, _ = features.shape
    img_feat_dim = C  # used to initialize projection layer
    
    num_classes = 2
    model = TopoRET(num_classes=num_classes, img_feat_dim=img_feat_dim, embed_dim=128)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 100
    
    train_model(model, train_loader, num_epochs=num_epochs, device=device)
    # test
    
    model.eval()
    all_labels = []
    all_probs = [] 
    all_preds = []
    
    with torch.no_grad():
        for images, PH_feats, labels in test_loader:
            images = images.to(device)
            PH_feats = PH_feats.to(device)
            labels = labels.to(device)
            outputs = model(images, PH_feats)  
            probs = torch.softmax(outputs, dim=1)  
            preds = torch.argmax(probs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    roc_auc = roc_auc_score(all_labels, all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)  
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp)
    
    pr_auc = average_precision_score(all_labels, all_probs)
    
    print("Test ROC-AUC: {:.4f}".format(roc_auc))
    print("Test Accuracy: {:.4f}".format(accuracy))
    print("Test Sensitivity (Recall): {:.4f}".format(recall))
    print("Test Specificity: {:.4f}".format(specificity))
    print("Test F1 Score: {:.4f}".format(f1))
    print("Test PR-AUC: {:.4f}".format(pr_auc))

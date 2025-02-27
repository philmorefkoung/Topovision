### Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
import pandas as pd
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, matthews_corrcoef)

# 1. Topo-RET 

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
        
        # Project image feature channels to embed_dim if needed.
        if img_feat_dim != embed_dim:
            self.img_proj = nn.Linear(img_feat_dim, embed_dim)
        else:
            self.img_proj = nn.Identity()
        
        # Embed each ph vector scalar into a vector of size embed_dim.
        self.ph_embed = nn.Linear(1, embed_dim)

        # normalize embedded PH tokens
        self.ph_norm = nn.LayerNorm(embed_dim)
        self.ph_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=3, batch_first=True),
            num_layers=3
        )
        self.ph_pool_attn = nn.Linear(embed_dim, 1)

        self.cross_attn_img = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.cross_attn_ph = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(5 * embed_dim, 3 * embed_dim),
            nn.ReLU(),

            nn.Linear(3 * embed_dim, 3 * embed_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(3 * embed_dim, num_classes)

    def forward(self, image, ph):
        """
          image: tensor of shape (B, 3, 128, 128)
          ph: tensor of shape (B, 400)
        """
        # Process image with Swin V2
        img_features = self.swin.forward_features(image)  # (B, C, H, W)
        B, C, H, W = img_features.shape
        img_tokens = img_features.flatten(2).transpose(1, 2)  # (B, N, C); N = H*W
        img_tokens = self.img_proj(img_tokens)  # (B, N, embed_dim)
        
        ph_tokens = self.ph_embed(ph.unsqueeze(-1))  # (B, 400, embed_dim)
        ph_tokens = self.ph_norm(ph_tokens)
        ph_tokens = self.ph_transformer(ph_tokens)
        
        ph_weights = torch.softmax(self.ph_pool_attn(ph_tokens), dim=1)  # (B, 400, 1)
        ph_weighted = (ph_tokens * ph_weights).sum(dim=1)  # (B, embed_dim)
        

        # Cross Attention 
        img_attended, _ = self.cross_attn_img(query=img_tokens, key=ph_tokens, value=ph_tokens)

        # Pooling 

        img_attended_mean = img_attended.mean(dim=1)
        img_attended_max = img_attended.max(dim=1)[0]
        img_attended_pooled = torch.cat([img_attended_mean, img_attended_max], dim=-1)  # (B, 2*embed_dim)

        img_tokens_mean = img_tokens.mean(dim=1)
        img_tokens_max = img_tokens.max(dim=1)[0]
        img_tokens_pooled = torch.cat([img_tokens_mean, img_tokens_max], dim=-1)  # (B, 2*embed_dim)

        # Fusion MLP
        # Concatenate all pooled representations.
        fused_features = torch.cat([img_attended_pooled,  img_tokens_pooled, ph_weighted], dim=-1) 

        fused = self.fusion_mlp(fused_features)  # (B, embed_dim)
        
        # Classification
        logits = self.classifier(fused)  # (B, num_classes)
        return logits

# 2. Dataset

class ImageNPZphDataset(Dataset):
    def __init__(self, npz_file, ph_file, transform=None):

        # images 
        npz_data = np.load(npz_file)
        self.images = npz_data['images']  # expected: (N, 3, 128, 128)
        
        # ph Vectors data.
        self.ph_data = pd.read_ph(ph_file)
        
        # Extract ph vectors 
        feature_cols = [str(i) for i in range(400)]
        self.ph_features = self.ph_data[feature_cols].values.astype(np.float32)
        
        # labels
        self.labels = self.ph_data['label'].values.astype(np.int64)
        
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #  image.
        image = self.images[idx]
        # convert image to a torch tensor if isn't
        if not torch.is_tensor(image):
            image = torch.tensor(image, dtype=torch.float)
        
        # permute to (C, H, W)
        if image.ndim == 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        
        # Apply transform 
        if self.transform:
            image = self.transform(image)
        
        # Get betti vectors and label.
        ph_feat = torch.tensor(self.ph_features[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, ph_feat, label

# 3. Data Split + Dataloaders

def create_dataloaders(npz_file, ph_file, batch_size=64, seed=18, transform=None):
    # Create full dataset
    full_dataset = ImageNPZphDataset(npz_file, ph_file, transform=transform)
    
    torch.manual_seed(seed)
    
    train_dataset, test_dataset = train_test_split(full_dataset, test_size=0.2, random_state=seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader


# 4. Training Loop

def train_model(model, train_loader, num_epochs, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, ph_feats, labels in train_loader:
            images = images.to(device)
            ph_feats = ph_feats.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, ph_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        scheduler.step()

if __name__ == '__main__':
    npz_file = '/path/to/images'
    ph_file = '/path/to/ph_vectors/and/labels'
    

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    train_loader, test_loader = create_dataloaders(
        npz_file, ph_file, batch_size=64, transform=transform
    )
    
    # Create dummy image (1, 3, 128, 128) to match input.
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
    img_feat_dim = C  #to initialize projection layer
    
    num_classes = 8
    model = TopoRET(num_classes=num_classes, img_feat_dim=img_feat_dim, embed_dim=96)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 100
    
    # Train
    train_model(model, train_loader, num_epochs=num_epochs, device=device)
    
    criterion = nn.CrossEntropyLoss()

    # Test
    model.eval()
    all_labels = []
    all_probs = [] 
    all_preds = []
    with torch.no_grad():
        for images, ph_feats, labels in test_loader:
            images = images.to(device)
            ph_feats = ph_feats.to(device)
            labels = labels.to(device)
            outputs = model(images, ph_feats)
            
            loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    mcc = matthews_corrcoef(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    
    # Report metrics.
    print(f'ROC-AUC: {roc_auc:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Weighted F1 Score: {weighted_f1:.4f}')
    print(f'Matthews Correlation Coefficient: {mcc:.4f}')
    print(f'Balanced Accuracy: {balanced_acc:.4f}')
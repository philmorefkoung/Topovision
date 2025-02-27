import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_dim = 400
num_classes = 8
batch_size = 64
epochs = 100
learning_rate = 1e-4

data = pd.read_csv('/path/to/betti/vectors')

X = data.drop(columns=['label'], axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values.squeeze(), dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32), torch.tensor(y_test.values.squeeze(), dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  
        return self.fc(x)

model = TransformerClassifier(input_dim, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

# Training 
for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_acc = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # metrics
        train_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss/len(train_loader):.4f}")
    scheduler.step()


model.eval()
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

all_labels = torch.cat(all_labels).numpy()   
all_preds  = torch.cat(all_preds).numpy()      
all_probs  = torch.cat(all_probs).numpy()      

# compute metrics
acc = accuracy_score(all_labels, all_preds)
weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
mcc = matthews_corrcoef(all_labels, all_preds)
balanced_acc = balanced_accuracy_score(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

print(f"\nTest Accuracy: {acc:.4f}")
print(f"Test Weighted F1: {weighted_f1:.4f}")
print(f"Test MCC: {mcc:.4f}")
print(f"Test Balanced Accuracy: {balanced_acc:.4f}")
print(f"Test ROC-AUC: {roc_auc:.4f}")
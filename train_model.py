import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import tqdm
import os

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUTS_FILE = 'inputs.npz'
TARGETS_FILE = 'targets.npz'
MODEL_SAVE_BEST = 'piyush_clone.pth'
MODEL_SAVE_FINAL = 'piyush_clone_final.pth'

# --- Dataset ---
class ChessDataset(Dataset):
    def __init__(self):
        print("Loading data...")
        # Load data into RAM (CPU)
        with np.load(INPUTS_FILE) as data:
            self.inputs = data['arr_0'].astype(np.float32)
        with np.load(TARGETS_FILE) as data:
            self.targets = data['arr_0'].astype(np.int64)
        
        print(f"Loaded {len(self.inputs)} samples.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# --- Model ---
class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        # Input: 12 x 8 x 8
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Flatten: 128 * 8 * 8 = 8192
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(8192, 1024)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 4096) # Output: 0-4095 classes

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- Training ---
def main():
    if not os.path.exists(INPUTS_FILE) or not os.path.exists(TARGETS_FILE):
        print("Data files not found!")
        return

    print(f"Using device: {DEVICE}")

    # 1. Prepare Data
    full_dataset = ChessDataset()
    
    # Split 90/10
    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # DataLoaders
    # pin_memory=True speeds up CPU->GPU transfer
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    # 2. Setup Model
    model = ChessModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            loop.set_postfix(loss=loss.item())
            
        train_acc = 100 * train_correct / train_total
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} Acc={train_acc:.2f}% | Val Loss={avg_val_loss:.4f} Acc={val_acc:.2f}%")
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_BEST)
            print(f"-> Saved new best model: {avg_val_loss:.4f}")

    # 4. Save Final
    torch.save(model.state_dict(), MODEL_SAVE_FINAL)
    print(f"Training Complete. Final model saved to {MODEL_SAVE_FINAL}")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

class CancerDataset(Dataset):
    def __init__(self):
        data = load_breast_cancer()
        X = data.data
        y = data.target

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
if __name__ == "__main__":
    torch.manual_seed(42)

    # 1. init dataset
    full_dataset = CancerDataset()

    # 2. spllit (80% train, 20% validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 3. Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle =True)
    val_loader = DataLoader(val_dataset,batch_size=32, shuffle=False)

    print(f"Dataset Size: {len(full_dataset)}")
    print(f"Training Batches: {len(train_loader)}")
    print(f"Validation Batches: {len(val_loader)}")

    # 4. Model setup
    model = Classifier(input_dim=30)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.BCELoss()

    num_epochs = 20
    print("\n---Training Start---")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for inputs, targets in train_loader:
            # A. forward pass
            preds = model(inputs)
            loss = criterion(preds, targets)

            # B. backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                preds = model(inputs)

                predicted_class = (preds > 0.5).float()
                correct += (predicted_class == targets).sum().item()

            acc = correct / len(val_loader.dataset)

            if (epoch +1) % 5 == 0:
                print(f"Epoch {epoch +1} | Loss: {avg_loss:.4f} | Valuation accuracy: {acc:.4f}")

    print("---Training Complete---")

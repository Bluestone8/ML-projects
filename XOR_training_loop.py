import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X_np, y_np = make_moons(n_samples=1000, noise=0.1, random_state=42)

X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}")

class MoonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
         nn.Linear(2, 16),
         nn.ReLU(),
         nn.Linear(16, 16),
         nn.ReLU(),
         nn.Linear(16,1),
         nn.Sigmoid()   
        )

    def forward(self, x):
        return self.network(x)
    
def train_one_epoch(model, inputs, targets, optimizer, criterion):
    model.train()

    preds = model(inputs)
    loss = criterion(preds, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    predicted_classes = (preds > 0.5).float()
    acc = (predicted_classes == targets).float().mean()
    
    return loss.item(), acc.item()

def evaluate(model, inputs, targets, criterion):
    model.eval() # set to evaluation mode (disables dropout/BatchNorm)

    with torch.no_grad():
        preds = model(inputs)
        loss = criterion(preds, targets)

        predicted_classes = (preds > 0.5).float()
        acc = (predicted_classes == targets).float().mean()

    return loss.item(), acc.item()

model = MoonModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

history = {'train_loss':[], 'val_loss':[], 'val_acc':[]}
best_acc = 0.0

print("---Starting Training---")
for epoch in range(500):
    t_loss, t_acc = train_one_epoch(model, X_train, y_train, optimizer, criterion)

    v_loss, v_acc = evaluate(model, X_val, y_val, criterion)

    history['train_loss'].append(t_loss)
    history['val_loss'].append(v_loss)
    history['val_acc'].append(v_acc)

    if v_acc > best_acc:
        best_acc = v_acc
        torch.save(model.state_dict(), "best_model.pth")
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Train Loss: {t_loss:.4f}  | Val acc: {v_acc:.4f}")

print(f"--- DONE. Best Val Acc: {best_acc:.4f} ---")
print("Model saved to 'best_model.pth'")

plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label = 'Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title("Learning Curve")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend()
plt.show()

import torch
import torch.nn as nn
import os
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from dataloader import create_data_loaders
from model import initialize_model, CustomEfficientNet

# Define hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create data loaders
train_dir = '/content/drive/MyDrive/POLYP_CLASSIFICATION_PROJECT/POLYP_DATA/m_train'
test_dir = '/content/drive/MyDrive/POLYP_CLASSIFICATION_PROJECT/POLYP_DATA/m_test'
valid_dir = '/content/drive/MyDrive/POLYP_CLASSIFICATION_PROJECT/POLYP_DATA/m_valid'

train_csv = os.path.join(train_dir, 'train.csv')
test_csv = os.path.join(test_dir, 'test.csv')
valid_csv = os.path.join(valid_dir, 'valid.csv')

train_loader, test_loader, valid_loader = create_data_loaders(train_dir, test_dir, valid_dir, batch_size)

# Calculate class weights
class_counts = [0, 0]
for _, labels in train_loader:
    unique, counts = torch.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        class_counts[u] += c.item()

class_weights = [sum(class_counts) / c for c in class_counts]
print(class_weights)
class_weights = torch.tensor(class_weights).to(device)

# Initialize the model
num_classes = 2  # Binary classification
model = initialize_model(num_classes)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights) #Cross-Entropy Loss with class weights
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).long()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)
        loss.backward()

        # Optimize
        optimizer.step()

        running_loss += loss.item()

    # Print the average loss for this epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), 'custom_efficientnet.pth')
print("Model saved.")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import create_data_loaders
from model import initialize_model, CustomEfficientNet

# Define hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create data loaders
data_dir = 'POLYP_DATA'
train_csv = 'm_train/train.csv'
test_csv = 'm_test/test.csv'
valid_csv = 'm_valid/valid.csv'

train_loader, test_loader, valid_loader = create_data_loaders(data_dir, train_csv, test_csv, valid_csv, batch_size)

# Initialize the model
num_classes = 2  # Binary classification
model = initialize_model(num_classes)
model.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels.float().view(-1, 1))  # Convert labels to float and reshape
        loss.backward()

        # Optimize
        optimizer.step()

        running_loss += loss.item()

    # Print the average loss for this epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), 'custom_efficientnet.pth')
print("Model saved.")


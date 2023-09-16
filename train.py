import torch
import torch.nn as nn
import os
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from dataloader import create_data_loaders
from model import initialize_model, CustomEfficientNet
import matplotlib.pyplot as plt
import numpy as np

# Define hyperparameters
num_epochs = 15
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

# Lists to track training and validation losses and accuracies
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

#best_valid_loss = float('inf')
best_valid_accuracy = float ('inf')
patience = 10  # Number of epochs with no improvement to wait
epochs_without_improvement = 0

#Helper function to compute accuracy
def compute_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    corrects = torch.sum(preds == labels)
    return corrects.item() / len(labels)

# Training loop
# Training loop
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    train_corrects = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).long()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)
        running_train_loss += loss.item()

        # Compute training accuracy within the loop
        train_corrects += compute_accuracy(outputs, labels) * len(labels)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

    train_loss = running_train_loss / len(train_loader)
    train_losses.append(train_loss)

    train_accuracy = train_corrects / len(train_loader.dataset)
    train_accuracies.append(train_accuracy)


    # Validation loss
    model.eval()
    running_valid_loss = 0.0
    valid_corrects = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images)

            # Calculate validation loss
            loss = criterion(outputs, labels)
            running_valid_loss += loss.item()

            # Compute validation accuracy within the loop
            valid_corrects += compute_accuracy(outputs, labels) * len(labels)

    valid_loss = running_valid_loss / len(valid_loader)
    valid_losses.append(valid_loss)

    valid_accuracy = valid_corrects / len(valid_loader.dataset)
    valid_accuracies.append(valid_accuracy)

    # Print the average losses and accuracies for this epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss} Valid Loss: {valid_loss} Train Acc: {train_accuracy} Valid Acc: {valid_accuracy}")

    # Inside your training loop, after computing the validation loss:
    if valid_accuracy < best_valid_accuracy:
        best_valid_accuracy = valid_accuracy
        torch.save(model.state_dict(), 'best_efficientnet.pth')
        print(f"Best model saved with validation {best_valid_accuracy} accuracy:")
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping due to no improvement in validation accuracy.")
            break

    # Enhanced Plotting of the training and validation losses and accuracies
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Twin the axes
    ax2 = ax1.twinx()

    # Plotting loss (on the left y-axis)
    ln1 = ax1.plot(train_losses, color='b', label='Training Loss')
    ln2 = ax1.plot(valid_losses, color='r', linestyle='--', label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(None)

    # Highlighting the epoch with the minimum validation loss
    max_valid_epoch = np.argmax(valid_accuracies)
    ax1.axvline(max_valid_epoch, color='gray', linestyle='--')
    ax1.annotate(f'Highest Valid accuracy at Epoch {max_valid_epoch + 1}', xy=(max_valid_epoch, valid_losses[max_valid_epoch]),
                xytext=(max_valid_epoch - 1, valid_losses[max_valid_epoch] + 0.05), arrowprops=dict(facecolor='black', arrowstyle='->'))

    # Plotting accuracy (on the right y-axis)
    ln3 = ax2.plot(train_accuracies, color='g', label='Training Accuracy')
    ln4 = ax2.plot(valid_accuracies, color='m', linestyle='--', label='Validation Accuracy')
    ax2.set_ylabel('Accuracy', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.grid(None)

    # Combined legend for both y-axes
    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    # Setting title
    plt.title('Training and Validation Losses & Accuracies')

    # Save the figure as an image
    plt.savefig('training_validation_plot.png')

    plt.draw()
# Display the plot
plt.show()

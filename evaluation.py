import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataloader import create_data_loaders
from model import initialize_model, CustomEfficientNet

# Define hyperparameters
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
try:
    model = initialize_model(num_classes=2)
    model.load_state_dict(torch.load("custom_efficientnet.pth"))
except FileNotFoundError:
    raise FileNotFoundError("Trained model file not found.")

#Ensure model is in device
model.to(device)

# Set the model to evaluation mode
model.eval()

# Create a data loader for the test dataset
data_dir = 'POLYP_DATA' 
test_csv = 'm_test/test.csv'

test_loader = DataLoader(
    PolypDataset(data_dir=data_dir, csv_file=test_csv, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])),
    batch_size=batch_size,
    shuffle=False  # Set to False for evaluation
)

# Initialize lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Evaluate the model on the test dataset
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)

        # Convert the outputs to binary predictions (0 or 1)
        predicted = (outputs > 0.5).float()

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

# Compute evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


class_names = ['0', '1']  # Class 0 (Non-adenomatous)', 'Class 1 (Adenomatous)
class_report = classification_report(true_labels, predicted_labels, target_names=class_names)
print("Per-Class Metrics:")
print(class_report)
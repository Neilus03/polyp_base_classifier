import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class PolypDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        """
        Args:
            data_dir (str): Path to the data directory containing 'images' and 'masks' subdirectories.
            csv_file (str): Path to the CSV file with image file names and labels.
            transform (callable, optional): Optional transform to be applied to the image.
        """
        self.data_dir = os.path.join(data_dir, 'images')  # Point to the 'images' subdirectory
        self.data_df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, str(self.data_df.iloc[idx, 0]) + '.tif')
        image = Image.open(img_name)
        label = int(self.data_df.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

def create_data_loaders(train_dir, test_dir, valid_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(PolypDataset(data_dir=train_dir, csv_file=os.path.join(train_dir, 'train.csv'), transform=transform), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(PolypDataset(data_dir=test_dir, csv_file=os.path.join(test_dir, 'test.csv'), transform=transform), batch_size=batch_size)
    valid_loader = DataLoader(PolypDataset(data_dir=valid_dir, csv_file=os.path.join(valid_dir, 'valid.csv'), transform=transform), batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, valid_loader




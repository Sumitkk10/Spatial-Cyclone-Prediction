import os
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

sequence_length = 48         # 2-day (48 hours) window
num_features_per_image = 128 # Number of features extracted per variable (adjust based on CNN architecture)
num_variables = 9            # Total number of variables
image_size = (64, 64)        # Resize images for consistent feature extraction
prediction_window = 48       # Next 24-hour prediction

# Paths to cyclone data folders
cyclone_data_folders = {
    "Cyclone_Fani": "dataset/train/Cyclone_Fani",
    "Cyclone_Amphan": "dataset/train/Cyclone_Amphan",
    "Cyclone_Asani": "dataset/train/Cyclone_Asani",
    "Cyclone_Yaas": "dataset/train/Cyclone_Yaas",
    "Cyclone_Jawad": "dataset/train/Cyclone_Jawad"
}

# Define cyclone-specific date ranges (assumed format is YYYY-MM-DD)
cyclone_date_ranges = {
    "Cyclone_Fani": ("2019-04-26", "2019-05-04"),
    "Cyclone_Amphan": ("2020-05-16", "2020-05-21"),
    "Cyclone_Asani": ("2022-05-08", "2022-05-12"),
    "Cyclone_Yaas": ("2021-05-23", "2021-05-26"),
    "Cyclone_Jawad": ("2021-12-02", "2021-12-06")
}

test_cyclone_date_ranges = {
    "Cyclone_Mandous": ("2022-12-09", "2022-12-12"),
    "Cyclone_Remal": ("2024-05-24", "2024-05-28")
}

test_data_folders = {
    "Cyclone_Mandous": "dataset/test/Cyclone_Mandous",
    "Cyclone_Remal": "dataset/test/Cyclone_Remal"
}

# Preprocessing transformation for images
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization
])

def load_images_from_directory(directory, variable):
    images = []
    for file_path in sorted(glob.glob(os.path.join(directory, f'*_{variable}.png'))):
        image = Image.open(file_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
        images.append(image)
    return images

# Load data for each cyclone and variable
def load_all_variable_data(cyclone_folder):
    data_by_variable = {}
    for variable in ["PS", "Q", "RH", "T", "TPREC", "TROPPB", "TS", "U", "V"]:  # Add other variables as needed
        data_by_variable[variable] = load_images_from_directory(cyclone_folder, variable)
    return data_by_variable

# Extract CNN features for each hour across all variables
def extract_hourly_features(data_by_variable, cnn):
    num_hours = len(next(iter(data_by_variable.values())))  # Assume all variables have the same length
    hourly_data = []

    for hour in range(num_hours):
        hourly_features = []
        for variable, images in data_by_variable.items():
            image = images[hour].to(device)
            with torch.no_grad():
                features = cnn(image)  # Extract features for the current hour
                features = features.view(-1).cpu().numpy()  # Flatten the feature map
            hourly_features.append(features)
        hourly_data.append(np.concatenate(hourly_features))  # Concatenate features for all variables
    return np.array(hourly_data)  # Shape: (num_hours, total_features)

def assign_labels(directory, start_date, end_date, reference_variable="PS"):
    labels = []
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Use only files for the reference variable
    for file_path in sorted(glob.glob(os.path.join(directory, f'*_{reference_variable}.png'))):
        date_str = os.path.basename(file_path).split('_')[0]  # Extract date from filename
        date = datetime.strptime(date_str, "%Y-%m-%d")
        label = 1 if start_date <= date <= end_date else 0
        labels.append(label)
    
    return np.array(labels)

class CyclonePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CyclonePredictor, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (hidden, _) = self.rnn(x)
        out = self.fc(hidden[-1])  # Only last hidden state for classification
        return torch.sigmoid(out)

# Function to create sequences with a specified length and next 24-hour labels
def create_sequences(hourly_data, labels, sequence_length, prediction_window=48):
    X, y = [], []
    for i in range(len(hourly_data) - sequence_length - prediction_window + 1):  # Ensure enough data for prediction window
        X.append(hourly_data[i:i + sequence_length])  # Sequence of features for `sequence_length` hours
        y.append(labels[i + sequence_length:i + sequence_length + prediction_window])  # Labels for the next 24 hours
    return np.array(X), np.array(y)

# Function to test the model on a single test cyclone and accumulate results
def test_model_on_cyclone(test_cyclone_name, test_folder, start_date, end_date):
    print(f"Testing on {test_cyclone_name}...")

    # Load images and labels
    data_by_variable = load_all_variable_data(test_folder)
    hourly_data = extract_hourly_features(data_by_variable, cnn)
    
    # Assign labels based on the cyclone active date range
    labels = assign_labels(test_folder, start_date, end_date)

    # Create sequences and labels for RNN
    X_test, y_test = create_sequences(hourly_data, labels, sequence_length, prediction_window)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).squeeze()
        predicted_labels = (predictions > 0.5).float()  # Threshold for binary classification
        
        # Calculate and print accuracy for first 12 hours and next 12 hours
        first_24_accuracy = accuracy_score(y_test_tensor[:, :24].cpu().numpy().flatten(),
                                           predicted_labels[:, :24].cpu().numpy().flatten())
        next_24_accuracy = accuracy_score(y_test_tensor[:, 24:].cpu().numpy().flatten(),
                                          predicted_labels[:, 24:].cpu().numpy().flatten())
        
        print(f"Accuracy on {test_cyclone_name} (First 24 hours): {first_24_accuracy * 100:.2f}%")
        print(f"Accuracy on {test_cyclone_name} (Next 24 hours): {next_24_accuracy * 100:.2f}%")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pre-trained CNN (ResNet) for feature extraction
    cnn = models.resnet18(pretrained=True)
    cnn = nn.Sequential(*list(cnn.children())[:-1])  # Remove last layer for feature extraction
    cnn.eval()
    cnn.to(device)  # Move the CNN model to the GPU if available

    # Define model hyperparameters
    hidden_size = 64
    num_layers = 2
    output_size = prediction_window  # Predict for the next 24 hours

    # Initialize RNN model
    model = CyclonePredictor(4 * num_variables * num_features_per_image, hidden_size, num_layers, output_size)
    model.to(device)  # Move the RNN model to the GPU
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model on each cyclone
    for cyclone_name, folder in cyclone_data_folders.items():
        print(f"Training on {cyclone_name}...")

        # Load images and labels
        data_by_variable = load_all_variable_data(folder)
        hourly_data = extract_hourly_features(data_by_variable, cnn)
        
        # Get date range for current cyclone
        start_date, end_date = cyclone_date_ranges[cyclone_name]
        labels = assign_labels(folder, start_date, end_date)

        # Create sequences and labels for RNN
        X_train, y_train = create_sequences(hourly_data, labels, sequence_length, prediction_window)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

        # Training loop
        num_epochs = 20
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train_tensor).squeeze()
            loss = criterion(output, y_train_tensor)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'next48.pth')
    print("Final model saved as 'next48.pth'.")

    # model.load_state_dict(torch.load('next48.pth'))
    # print("Model loaded for testing.")

    # Test the model on each test cyclone
    for test_cyclone_name, test_folder in test_data_folders.items():
        start_date, end_date = test_cyclone_date_ranges[test_cyclone_name]
        test_model_on_cyclone(test_cyclone_name, test_folder, start_date, end_date)

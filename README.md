# Spatial-Cyclone-Prediction
Early Detection and Prediction of Cyclones Using Spatial Data Analysis

This repository contains a Python-based deep learning project to predict cyclone activity for the next 24 hours using sequential weather data from satellite images. The model uses CNN feature extraction followed by an LSTM network to predict cyclone presence in the upcoming hours, focusing on various critical weather variables.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)
- [Contributing](#contributing)

## Overview

This project aims to predict cyclone activity by analyzing satellite images across multiple meteorological variables. Each variable has hourly images collected over time, which are used as input for a Convolutional Neural Network (CNN) that extracts features. These features are then passed to a Long Short-Term Memory (LSTM) model to perform time series forecasting for cyclone prediction in the next 24 hours. The model is trained and evaluated on individual cyclones and tested on unseen cyclones to assess generalization.

## Dataset

### Cyclone Data
Each cyclone dataset contains images for 9 meteorological variables, saved hourly:
- **PS** - Surface Pressure (Pa)
- **Q** - Specific Humidity (Mass fraction)
- **RH** - Relative Humidity
- **T** - Air Temperature (K)
- **TPREC** - Total Precipitation (kg m^2 / s)
- **TROPPB** - Tropopause pressure based on blended estimate (Pa)
- **TS** - Surface skin temperature (K)
- **U** - Eastward Wind (m/s)
- **V** - Northward Wind (m/s)

## Requirements

The following dependencies are required:
- Python 3.8+
- PyTorch
- torchvision
- scikit-learn
- Pillow
- NumPy

Install dependencies via:
```bash
pip install -r requirements.txt
```

## Usage

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-url>
   ```

2. **Install Dependencies Install the required Python packages with:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run Training and Testing To train and test the model, execute the main script:**
    ```bash
    python main.py
    ```

During training, the model will:

    * Extract features from the satellite images of each cyclone.
    * Train the LSTM model using the extracted features.
    * Display the training loss after each epoch.

4. **Configure Hyperparameters (Optional) Adjust any model parameters, file paths, or other settings in main.py if necessary.**

## Testing
After training, the model will be evaluated on test cyclones provided in dataset/test/. For each test cyclone, the accuracy will be printed separately for the first 12 hours and the next 12 hours of the prediction window.

## Results
The results are evaluated based on the accuracy of cyclone prediction in the next 24-hour window. Performance is calculated separately for the first 12 hours and the second 12 hours, giving insights into how the model performs as prediction time increases.

## Contributing
If you’d like to contribute to this project, please fork the repository, make your changes, and submit a pull request.
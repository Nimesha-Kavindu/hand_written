# Handwritten Character Recognition System

A complete handwritten digit and letter recognition system using K-Nearest Neighbors (KNN) algorithm with an interactive Tkinter GUI.

## Features

- **KNN Classifier**: Uses scikit-learn's KNeighborsClassifier for character recognition
- **MNIST Dataset**: Supports digits (0-9) and can be extended to letters (A-Z) using EMNIST
- **Interactive GUI**: Draw characters using mouse on a canvas
- **Real-time Prediction**: Get instant predictions with confidence scores
- **Adjustable Brush Size**: Customize drawing brush size
- **Model Training Notebooks**: Complete workflow from data loading to model training

## Project Structure

```
hand_written/
├── notebooks/
│   ├── 01_data_loading.ipynb      # Load and preprocess MNIST/EMNIST data
│   └── 02_model_training.ipynb    # Train KNN model and evaluate performance
├── data/                          # Preprocessed datasets (created after running notebooks)
├── models/                        # Trained models (created after training)
├── gui_app.py                     # Tkinter GUI application
└── README.md                      # This file
```

## Installation

1. Install required packages:
```bash
pip install numpy matplotlib scikit-learn pillow seaborn
```

## Usage

### Step 1: Load and Preprocess Data

Open and run `notebooks/01_data_loading.ipynb`:
- Loads MNIST dataset (digits 0-9)
- Optionally loads EMNIST dataset (letters A-Z)
- Normalizes and preprocesses images
- Splits data into training and test sets
- Saves preprocessed data to `data/preprocessed_data.pkl`

### Step 2: Train the Model

Open and run `notebooks/02_model_training.ipynb`:
- Loads preprocessed data
- Trains KNN models with different k values
- Evaluates model performance
- Visualizes predictions and confusion matrix
- Saves the best model to `models/knn_model.pkl`

### Step 3: Run the GUI Application

```bash
python gui_app.py
```

## Using the GUI

1. **Draw**: Click and drag on the white canvas to draw a digit or letter
2. **Adjust Brush**: Use the slider to change brush size
3. **Predict**: Click "Predict" to get the model's prediction
4. **Clear**: Click "Clear" to erase the canvas and start over

The prediction will show:
- The predicted character
- Confidence percentage
- Top 3 predictions with probabilities

## Model Performance

The KNN model typically achieves:
- **Digits (0-9)**: ~95-97% accuracy on MNIST
- Training time depends on dataset size and k value
- Optimal k value is determined through cross-validation

## Customization

### Adjust Training Data Size

In `02_model_training.ipynb`, modify:
```python
train_samples = 10000  # Increase for better accuracy
test_samples = 2000
```

### Change K Value

The notebook automatically finds the best k value, but you can manually set it:
```python
best_k = 5  # Your preferred k value
```

### Add Letter Recognition

Uncomment and configure the EMNIST loading section in `01_data_loading.ipynb` to include letters A-Z.

## Troubleshooting

**Model not found error**: Make sure to run the training notebook first to generate the model file.

**Low accuracy**: Try increasing the training samples or adjusting the k value.

**Slow predictions**: Reduce training dataset size or use a smaller k value.

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- scikit-learn
- Pillow (PIL)
- Seaborn
- Tkinter (usually included with Python)

## License

This project is open source and available for educational purposes.

## Future Improvements

- Add support for multiple character recognition
- Implement neural network models (CNN)
- Add data augmentation techniques
- Export predictions to file
- Add model comparison features

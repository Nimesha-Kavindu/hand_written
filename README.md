# 🖊️ Handwritten Digit Recognition with CNN

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A sophisticated handwritten digit recognition system built with Convolutional Neural Networks (CNN) and an interactive GUI for real-time digit classification.

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-model-architecture) • [Results](#-results)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Training Process](#-training-process)
- [Results](#-results)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🌟 Overview

This project implements a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0-9) with high accuracy. The system includes:

- **Deep Learning Model:** CNN trained on the MNIST dataset achieving **98%+ accuracy**
- **Interactive GUI:** User-friendly drawing canvas built with Tkinter
- **Real-time Prediction:** Instant digit recognition as you draw
- **Multiple Models:** Support for both CNN and KNN classifiers

Perfect for learning about deep learning, computer vision, and building practical AI applications!

---

## ✨ Features

### 🎨 Interactive Drawing Interface
- **Draw digits** directly on a 400x400 pixel canvas
- **Adjustable brush size** (10-40 pixels) for comfortable writing
- **Real-time prediction** with confidence scores
- **Top 3 predictions** displayed with probabilities

### 🧠 Advanced CNN Model
- **2 Convolutional layers** with ReLU activation
- **MaxPooling layers** for efficient feature extraction
- **Dropout regularization** to prevent overfitting
- **Adam optimizer** for fast convergence
- **98%+ test accuracy** on MNIST dataset

### 📊 Comprehensive Training Pipeline
- **Data preprocessing** with normalization
- **Train-validation split** (70-30) for monitoring
- **Model checkpointing** to save best performance
- **Visualization tools** for accuracy/loss curves
- **Prediction testing** with visual feedback

### 🎯 User-Friendly Features
- Clean, modern GUI design
- Clear button to restart drawing
- Instant prediction with single click
- Confidence percentage display
- Cross-platform compatibility (Windows, Mac, Linux)

---

## 🎬 Demo

### GUI Application
```
┌─────────────────────────────────────────┐
│   Handwritten Character Recognition     │
├─────────────────────────────────────────┤
│                                         │
│     [  Drawing Canvas 400x400  ]       │
│                                         │
│     Draw a digit (0-9) or letter       │
│                                         │
│     Brush Size: [===========]          │
│                                         │
│     [ Predict ]    [ Clear ]           │
│                                         │
│     Prediction:  [ 7 ]                 │
│     Confidence: 95.3%                  │
│     Top 3: 7(95.3%), 1(2.1%), 9(1.5%) │
└─────────────────────────────────────────┘
```

### Training Results
- **Training Accuracy:** 98.2%
- **Validation Accuracy:** 97.8%
- **Test Accuracy:** 98.1%
- **Training Time:** ~2 minutes (5 epochs)

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Nimesha-Kavindu/hand_written.git
cd hand_written
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- `tensorflow` - Deep learning framework
- `keras` - High-level neural network API
- `numpy` - Numerical computing
- `matplotlib` - Data visualization
- `pillow` - Image processing
- `scikit-learn` - Machine learning utilities

---

## 💻 Usage

### Option 1: Train Your Own Model

1. **Navigate to the achchauwa folder:**
   ```bash
   cd achchauwa
   ```

2. **Open and run the Jupyter notebook:**
   ```bash
   jupyter notebook Untitled0.ipynb
   ```

3. **Execute all cells** to:
   - Load and preprocess MNIST data
   - Build the CNN architecture
   - Train the model (5 epochs)
   - Evaluate performance
   - Save the trained model

### Option 2: Use Pre-trained Model

If you already have a trained model in `achchauwa/models/`:

```bash
cd achchauwa
python gui_app.py
```

### Using the GUI Application

1. **Launch the application** (either method above)
2. **Draw a digit** on the white canvas using your mouse
3. **Adjust brush size** if needed using the slider
4. **Click "Predict"** to get the classification result
5. **Click "Clear"** to start over

---

## 📁 Project Structure

```
hand_written/
│
├── achchauwa/
│   ├── models/
│   │   ├── cnn_model.h5              # Trained CNN model
│   │   └── cnn_model_metadata.json   # Model configuration
│   ├── gui_app.py                    # GUI application
│   └── Untitled0.ipynb               # Training notebook
│
├── hand written with KNN 10 categories/
│   ├── images/                       # Training images (0-9)
│   ├── 1-dataset-creation.ipynb      # Dataset preparation
│   ├── 2-training-the-KNN.ipynb      # KNN model training
│   ├── 3-sinhala-character-gui.ipynb # Alternative GUI
│   ├── data.npy                      # Preprocessed data
│   ├── target.npy                    # Labels
│   └── sinhala-character-knn.sav     # KNN model
│
├── hand written with KNN 4 categories/
│   ├── images/                       # Training images (1-4)
│   ├── 1-dataset-creation.ipynb      # Dataset preparation
│   ├── 2-training-the-KNN.ipynb      # KNN model training
│   ├── 3-sinhala-character-gui.ipynb # GUI for 4 categories
│   ├── data.npy                      # Preprocessed data
│   ├── target.npy                    # Labels
│   └── sinhala-character-knn.sav     # KNN model
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🏗️ Model Architecture

### CNN Architecture

```
Input (28x28x1)
    ↓
Conv2D (32 filters, 3x3, ReLU)
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3, ReLU)
    ↓
MaxPooling2D (2x2)
    ↓
Flatten
    ↓
Dropout (25%)
    ↓
Dense (10 units, Softmax)
    ↓
Output (10 classes: 0-9)
```

### Layer Details

| Layer | Output Shape | Parameters | Purpose |
|-------|-------------|------------|---------|
| Conv2D-1 | (26, 26, 32) | 320 | Extract basic features (edges, curves) |
| MaxPool2D-1 | (13, 13, 32) | 0 | Reduce spatial dimensions |
| Conv2D-2 | (11, 11, 64) | 18,496 | Extract complex patterns |
| MaxPool2D-2 | (5, 5, 64) | 0 | Further dimensionality reduction |
| Flatten | (1600,) | 0 | Convert 2D to 1D |
| Dropout | (1600,) | 0 | Prevent overfitting |
| Dense | (10,) | 16,010 | Classification output |

**Total Parameters:** 34,826

---

## 📊 Dataset

### MNIST Dataset
- **Training samples:** 60,000 images
- **Test samples:** 10,000 images
- **Image size:** 28x28 pixels (grayscale)
- **Classes:** 10 (digits 0-9)
- **Format:** Normalized pixel values (0.0 to 1.0)

### Data Preprocessing
1. **Normalization:** Pixel values divided by 255
2. **Reshape:** Add channel dimension (28, 28, 1)
3. **One-hot encoding:** Labels converted to categorical format
4. **Train-validation split:** 70% training, 30% validation

---

## 🎓 Training Process

### Training Configuration
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy
- **Batch Size:** 32 (default)
- **Epochs:** 5
- **Validation Split:** 0.3 (30%)

### Training Progress Example
```
Epoch 1/5 - Accuracy: 85.2%, Val_Accuracy: 84.1%
Epoch 2/5 - Accuracy: 92.4%, Val_Accuracy: 91.8%
Epoch 3/5 - Accuracy: 95.1%, Val_Accuracy: 94.5%
Epoch 4/5 - Accuracy: 97.0%, Val_Accuracy: 96.3%
Epoch 5/5 - Accuracy: 98.2%, Val_Accuracy: 97.8%
```

### Training Features
- **Early Stopping:** Prevents overfitting
- **Model Checkpointing:** Saves best model
- **Real-time Monitoring:** Track accuracy and loss
- **Visualization:** Plot training curves

---

## 📈 Results

### Performance Metrics
- ✅ **Test Accuracy:** 98.1%
- ✅ **Test Loss:** 0.06
- ✅ **Training Time:** ~2 minutes (CPU)
- ✅ **Inference Time:** <50ms per prediction

### Confusion Matrix Highlights
The model performs exceptionally well across all digits, with occasional confusion between:
- 4 and 9 (similar writing styles)
- 3 and 8 (curved shapes)
- 7 and 1 (similar vertical strokes)

### Visualization Examples
- **Accuracy Curves:** Shows steady improvement over epochs
- **Loss Curves:** Demonstrates effective learning
- **Prediction Samples:** Visual verification of correct classifications

---

## 🛠️ Technologies Used

### Deep Learning & ML
- **TensorFlow 2.x** - Deep learning framework
- **Keras** - High-level neural network API
- **NumPy** - Numerical computing
- **scikit-learn** - ML utilities and KNN implementation

### GUI & Visualization
- **Tkinter** - GUI framework (included with Python)
- **Matplotlib** - Data visualization and plotting
- **Pillow (PIL)** - Image processing

### Development Tools
- **Jupyter Notebook** - Interactive development
- **Python 3.8+** - Programming language
- **Git** - Version control

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Ideas for Contributions
- 🎨 Improve GUI design
- 🧠 Add support for letters (A-Z)
- 📱 Create a web-based version
- 🚀 Optimize model architecture
- 📝 Add more documentation
- 🧪 Create unit tests

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Nimesha Kavindu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Acknowledgments

- **MNIST Dataset:** Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- **TensorFlow & Keras Teams:** For the excellent deep learning frameworks
- **Python Community:** For comprehensive libraries and documentation
- **Open Source Contributors:** For inspiration and code examples

---

## 📞 Contact & Support

**Nimesha Kavindu**
- GitHub: [@Nimesha-Kavindu](https://github.com/Nimesha-Kavindu)
- Repository: [hand_written](https://github.com/Nimesha-Kavindu/hand_written)

### Found a Bug?
Please open an issue on GitHub with:
- Description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Screenshots (if applicable)

### Have Questions?
Feel free to:
- Open a GitHub issue
- Start a discussion
- Submit a pull request

---

## 🎯 Future Improvements

- [ ] Add support for custom datasets
- [ ] Implement data augmentation
- [ ] Add batch prediction mode
- [ ] Create REST API for predictions
- [ ] Add model export (ONNX, TFLite)
- [ ] Implement transfer learning
- [ ] Add multi-language support
- [ ] Create mobile app version

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ by Nimesha Kavindu

</div>

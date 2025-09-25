# Network Intrusion Detection System - Setup and Usage Guide

## 🎯 Project Overview

This project implements a Network Intrusion Detection System using Machine Learning and Deep Learning techniques on the NSL-KDD dataset. It includes data preprocessing, multiple classification algorithms, and comprehensive model evaluation.

## ✅ Setup Status

**✅ SYSTEM IS NOW FULLY FUNCTIONAL**

All required dependencies are installed and the system has been tested successfully:
- ✅ Data loading and preprocessing
- ✅ Machine learning model training  
- ✅ Deep learning with TensorFlow/Keras
- ✅ All notebooks are ready to run

## 📋 Prerequisites

- Python 3.13.3 (configured)
- All required packages installed:
  - numpy, pandas, scikit-learn
  - matplotlib, seaborn (visualization)
  - tensorflow, keras (deep learning)
  - jupyter notebook

## 🚀 Quick Start

### 1. Test the System
```bash
python test_system.py
```
This will verify all components are working correctly.

### 2. Run the Notebooks

Open VS Code and run the notebooks in this order:

#### Option A: Individual Notebooks (Recommended for learning)
1. **Data_Preprocessing_NSL_KDD.ipynb** - Data preprocessing and feature engineering
2. **Classifiers_NSL_KDD.ipynb** - Train and evaluate machine learning models
3. **Intrusion_Detection_NSL_KDD.ipynb** - Complete pipeline with deep learning

#### Option B: All-in-One Notebook
Run **Intrusion_Detection_NSL_KDD.ipynb** for the complete pipeline

## 📁 Project Structure

```
Network-Intrusion-Detection-Using-Machine-Learning/
├── 📓 Notebooks/
│   ├── Data_Preprocessing_NSL_KDD.ipynb      # Data preprocessing
│   ├── Classifiers_NSL_KDD.ipynb             # ML models
│   └── Intrusion_Detection_NSL_KDD.ipynb     # Complete pipeline
├── 📊 datasets/
│   ├── KDDTrain+.txt                         # Original NSL-KDD dataset
│   ├── bin_data.csv                          # Binary classification data
│   └── multi_data.csv                        # Multi-class classification data
├── 🏷️ labels/
│   ├── le1_classes.npy                       # Binary labels
│   └── le2_classes.npy                       # Multi-class labels
├── 🤖 models/                                # Trained models
├── 📈 plots/                                 # Generated visualizations
├── ⚖️ weights/                               # Neural network weights
├── 📄 requirements.txt                       # Dependencies
├── 🧪 test_system.py                         # System verification
└── 📚 README.md & documentation.md           # Documentation
```

## 🔬 Available Models

### Traditional Machine Learning:
- **Linear Support Vector Machine** (96.69% binary, 95.24% multi-class)
- **Quadratic Support Vector Machine** (95.71% binary, 92.86% multi-class)
- **K-Nearest Neighbors** (98.55% binary, 98.29% multi-class)
- **Linear Discriminant Analysis** (96.70% binary, 93.19% multi-class)
- **Quadratic Discriminant Analysis** (68.79% binary, 44.96% multi-class)

### Deep Learning:
- **Multi-Layer Perceptron** (97.79% binary, 96.92% multi-class)
- **Long Short-Term Memory (LSTM)** (83.05% binary)
- **Autoencoder** (92.26% binary, 91.22% multi-class)

## 📊 Dataset Information

- **Source**: NSL-KDD Dataset (Canadian Institute for Cybersecurity)
- **Size**: 125,973 records
- **Features**: 42 attributes (reduced from 43)
- **Attack Types**: 
  - Binary: Normal vs. Attack
  - Multi-class: Normal, DoS, Probe, R2L, U2R

## 🎯 Attack Classification

### Binary Classification (2 classes):
- **Normal**: Legitimate network traffic
- **Attack**: Any type of intrusion

### Multi-class Classification (5 classes):
- **Normal**: Legitimate traffic
- **DoS**: Denial of Service attacks
- **Probe**: Surveillance and probing attacks
- **R2L**: Remote to Local attacks
- **U2R**: User to Root attacks

## 🛠️ Data Preprocessing Pipeline

1. **Data Loading**: Load NSL-KDD dataset with proper column names
2. **Data Cleaning**: Remove unnecessary attributes (difficulty_level)
3. **Feature Engineering**:
   - Normalization using Standard Scaler
   - One-hot encoding for categorical variables
   - Feature selection using Pearson correlation
4. **Label Encoding**: Prepare targets for binary and multi-class classification
5. **Data Splitting**: 80/20 train-test split

## 📈 Model Evaluation Metrics

- **Accuracy Score**: Overall classification accuracy
- **Classification Report**: Precision, Recall, F1-score
- **Confusion Matrix**: Detailed prediction analysis
- **ROC Curves**: Binary classification performance
- **Training History**: Loss and accuracy plots for neural networks

## 🎨 Visualizations

The system generates comprehensive visualizations:
- Data distribution pie charts
- Model performance comparisons
- ROC curves for binary classification
- Training history plots
- Confusion matrices
- Real vs Predicted scatter plots

## 🚦 Running Instructions

### Method 1: VS Code Jupyter Extension (Recommended)
1. Open any notebook file (.ipynb)
2. VS Code will automatically detect and configure the kernel
3. Run cells sequentially or all at once (Ctrl+F9)

### Method 2: Command Line
```bash
# Navigate to project directory
cd "path/to/Network-Intrusion-Detection-Using-Machine-Learning-master"

# Start Jupyter Notebook
jupyter notebook

# Open desired notebook in browser
```

## 🔧 Troubleshooting

### Common Issues:

1. **Import Errors**: Ensure all packages are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Path Issues**: Make sure you're in the correct directory

3. **Memory Issues**: For large datasets, consider reducing batch sizes

4. **Kernel Issues**: Restart the kernel if cells aren't executing

## 📖 Learning Path

### For Beginners:
1. Start with `Data_Preprocessing_NSL_KDD.ipynb` to understand data preparation
2. Move to `Classifiers_NSL_KDD.ipynb` for traditional ML models
3. Finish with `Intrusion_Detection_NSL_KDD.ipynb` for deep learning

### For Advanced Users:
- Run `Intrusion_Detection_NSL_KDD.ipynb` for the complete pipeline
- Experiment with hyperparameter tuning
- Try different model architectures
- Implement additional evaluation metrics

## 🎓 Key Concepts Covered

- **Data Preprocessing**: Normalization, encoding, feature selection
- **Machine Learning**: SVM, KNN, LDA, QDA
- **Deep Learning**: MLP, LSTM, Autoencoders
- **Model Evaluation**: Cross-validation, metrics, visualization
- **Cybersecurity**: Network intrusion detection, attack classification

## 📚 References

1. **Research Paper**: "A Novel Statistical Analysis and Autoencoder Driven Intelligent Intrusion Detection Approach"
2. **Dataset**: NSL-KDD from Canadian Institute for Cybersecurity
3. **Medium Article**: [Network Intrusion Detection using Deep Learning](https://medium.com/geekculture/network-intrusion-detection-using-deep-learning-bcc91e9b999d)

## 🎉 Next Steps

Now that the system is working, you can:

1. **Explore the notebooks** to understand the methodology
2. **Experiment with different models** and hyperparameters
3. **Add new features** or preprocessing steps
4. **Try different datasets** or extend to real-time detection
5. **Deploy models** for production use

## 📞 Support

If you encounter any issues:
1. Check the `test_system.py` output for diagnostics
2. Ensure all files are present in their correct directories
3. Verify Python environment and package versions
4. Review the notebook outputs for error messages

---

**🚀 The system is ready to use! Start with any notebook and explore the fascinating world of network intrusion detection using machine learning.**
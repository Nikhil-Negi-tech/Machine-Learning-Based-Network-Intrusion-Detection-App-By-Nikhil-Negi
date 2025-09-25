#!/usr/bin/env python
"""
Network Intrusion Detection System Test Script
Test script to verify all components are working correctly
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

def test_data_loading():
    """Test if all required data files exist and can be loaded"""
    print("Testing data loading...")
    
    # Check if dataset files exist
    dataset_files = [
        'datasets/KDDTrain+.txt',
        'datasets/bin_data.csv', 
        'datasets/multi_data.csv',
        'labels/le1_classes.npy',
        'labels/le2_classes.npy'
    ]
    
    for file in dataset_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            return False
    
    try:
        # Test loading preprocessed data
        bin_data = pd.read_csv('datasets/bin_data.csv')
        multi_data = pd.read_csv('datasets/multi_data.csv')
        le1_classes = np.load('labels/le1_classes.npy', allow_pickle=True)
        le2_classes = np.load('labels/le2_classes.npy', allow_pickle=True)
        
        print(f"✓ Binary classification data shape: {bin_data.shape}")
        print(f"✓ Multi-class classification data shape: {multi_data.shape}")
        print(f"✓ Binary labels loaded: {len(le1_classes)} classes")
        print(f"✓ Multi-class labels loaded: {len(le2_classes)} classes")
        
        return True
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False

def test_model_training():
    """Test a simple model training pipeline"""
    print("\nTesting model training...")
    
    try:
        # Load binary data for quick test
        bin_data = pd.read_csv('datasets/bin_data.csv')
        bin_data.drop(bin_data.columns[0], axis=1, inplace=True)
        
        # Prepare data - drop non-numeric columns
        numeric_columns = bin_data.select_dtypes(include=[np.number]).columns
        X = bin_data[numeric_columns].drop('intrusion', axis=1, errors='ignore')
        y = bin_data['intrusion']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a simple SVM model
        svm_model = SVC(kernel='linear', C=1.0)
        svm_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✓ SVM model trained successfully")
        print(f"✓ Test accuracy: {accuracy:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Error in model training: {e}")
        return False

def test_tensorflow():
    """Test TensorFlow/Keras functionality"""
    print("\nTesting TensorFlow/Keras...")
    
    try:
        print(f"✓ TensorFlow version: {tf.__version__}")
        
        # Create a simple neural network
        try:
            from keras.models import Sequential
            from keras.layers import Dense
            model = Sequential([
                Dense(10, activation='relu', input_shape=(5,)),
                Dense(1, activation='sigmoid')
            ])
        except ImportError:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("✓ Neural network created and compiled successfully")
        
        return True
    except Exception as e:
        print(f"✗ Error with TensorFlow: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("NETWORK INTRUSION DETECTION SYSTEM - TEST SUITE")
    print("=" * 50)
    
    tests = [
        test_data_loading,
        test_model_training, 
        test_tensorflow
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The system is ready to use.")
        print("\nYou can now run the notebooks:")
        print("1. Data_Preprocessing_NSL_KDD.ipynb - For data preprocessing")
        print("2. Classifiers_NSL_KDD.ipynb - For machine learning models")
        print("3. Intrusion_Detection_NSL_KDD.ipynb - For complete pipeline")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()